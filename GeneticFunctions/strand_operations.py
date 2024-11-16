import pandas as pd
from joblib import Parallel, delayed
import os
from .aux_functions import *

def flip_strand(allele):
    """
    Flips a DNA base to its complementary base.
    
    Parameters:
    - allele (str): A single nucleotide base ('A', 'T', 'C', or 'G').
    
    Returns:
    - str or None: The complementary base if input is valid, otherwise None.
    """
    allele_map = {
        "A": "T",
        "C": "G",
        "T": "A",
        "G": "C"
    }

    return allele_map.get(allele, None)

def snp_match(sumstats, info_snp, strand_flip=True, join_by_pos=True, remove_dups=True,
              match_min_prop=0.2, return_flip_and_rev=False, from_file=False):
    """
    Matches SNPs from a summary statistics DataFrame with an info SNP DataFrame,
    handling strand flips, reversed alleles, and duplicate removal as specified.
    
    Parameters:
    - sumstats (pd.DataFrame or str): Summary statistics containing SNP data to be matched or Path to sumstats TSV file.
    - info_snp (pd.DataFrame or str): Information on SNPs to match against or plink prefix.
    - strand_flip (bool): Whether to allow strand flipping to match SNPs.
    - join_by_pos (bool): Whether to join by position (True) or by 'rsid' (False).
    - remove_dups (bool): Whether to remove duplicate matches.
    - match_min_prop (float): Minimum proportion of matches required.
    - return_flip_and_rev (bool): Whether to return columns indicating strand flips and reversals.
    - from_file (bool): Whether sumstats and info_snp are file paths (default=False).
    
    Returns:
    - pd.DataFrame: DataFrame with matched SNPs, sorted by chromosome and position.
    
    Raises:
    - ValueError: If from_file is True but inputs are not file paths (strings),
                  or If required columns are missing, or if matching SNPs does not meet the minimum proportion.
    """
    n_jobs = os.cpu_count() - 1
    print("Using {n_jobs} CPUs")
    if from_file:
        if not isinstance(sumstats, str) or not isinstance(info_snp, str):
            raise ValueError("When 'from_file' is True, 'sumstats' and 'info_snp' must be file paths (strings).")
        
        sumstats, info_snp = load_data_for_snp_match(sumstats, info_snp)

    info_snp = filter_standard_chromosomes(info_snp).copy()
    sumstats = filter_standard_chromosomes(sumstats).copy()
    
    sumstats['_NUM_ID_'] = range(len(sumstats))
    info_snp['_NUM_ID_'] = range(len(info_snp))

    min_match = match_min_prop * min(len(sumstats), len(info_snp))
    join_by = ['chr', 'pos' if join_by_pos else 'rsid', 'a0', 'a1']

    required_sumstats = set(join_by + ['beta'])
    required_info_snp = set(join_by + ['pos'])
    if not required_sumstats.issubset(sumstats.columns):
        raise ValueError(f"Please use proper names for variables in 'sumstats'. Expected {', '.join(required_sumstats)}.")
    if not required_info_snp.issubset(info_snp.columns):
        raise ValueError(f"Please use proper names for variables in 'info_snp'. Expected {', '.join(required_info_snp)}.")

    print(f"{len(sumstats):,} variants to be matched.")

    sumstats = sumstats[sumstats.set_index(join_by[:2]).index.isin(info_snp.set_index(join_by[:2]).index)]
    if len(sumstats) == 0:
        raise ValueError("No variant has been matched.")

    # Handle strand flipping in parallel
    if strand_flip:
        ambiguous_snps = sumstats[['a0', 'a1']].apply(lambda x: " ".join(x) in ["A T", "T A", "C G", "G C"], axis=1)
        print(f"{ambiguous_snps.sum():,} ambiguous SNPs have been removed.")
        
        sumstats2 = sumstats[~ambiguous_snps].copy()
        sumstats3 = sumstats2.copy()
        
        sumstats2['_FLIP_'] = False
        sumstats3['_FLIP_'] = True

        # Parallel strand flipping
        sumstats3['a0'] = Parallel(n_jobs=n_jobs)(delayed(flip_strand)(a) for a in sumstats2['a0'])
        sumstats3['a1'] = Parallel(n_jobs=n_jobs)(delayed(flip_strand)(a) for a in sumstats2['a1'])
        
        sumstats3 = pd.concat([sumstats2, sumstats3], ignore_index=True)
    else:
        sumstats3 = sumstats.copy()
        sumstats3['_FLIP_'] = False

    # Handle reversed alleles in parallel
    sumstats4 = sumstats3.copy()
    sumstats3['_REV_'] = False
    sumstats4['_REV_'] = True

    # Reverse alleles and negate beta in parallel
    sumstats4['a0'] = sumstats3['a1']
    sumstats4['a1'] = sumstats3['a0']
    sumstats4['beta'] = Parallel(n_jobs=n_jobs)(delayed(lambda beta: -beta)(b) for b in sumstats3['beta'])
    
    sumstats4 = pd.concat([sumstats3, sumstats4], ignore_index=True)

    # Matching by merging
    matched = pd.merge(sumstats4, info_snp, on=join_by, how='inner', suffixes=('.ss', ''))
    
    # Remove duplicates
    if remove_dups:
        dups = matched.duplicated(subset=['chr', 'pos'])
        if dups.any():
            matched = matched[~dups]
            print("Some duplicates were removed.")
    
    print(f"{len(matched):,} variants have been matched; {matched['_FLIP_'].sum():,} were flipped and {matched['_REV_'].sum():,} were reversed.")

    if len(matched) < min_match:
        raise ValueError("Not enough variants have been matched.")

    # Remove _FLIP_ and _REV_ columns if not required
    if not return_flip_and_rev:
        matched = matched.drop(columns=['_FLIP_', '_REV_'])

    return matched.sort_values(by=['chr', 'pos']).reset_index(drop=True)

