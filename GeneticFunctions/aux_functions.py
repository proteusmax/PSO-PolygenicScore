import pandas as pd
from pandas_plink import read_plink1_bin
import os
import numpy as np

def load_data_for_snp_match(sumstats_path, plink_prefix):
    """
    Reads the info_snp and sumstats TSV files into Pandas DataFrames
    and returns them in the correct format for the snp_match function.
    
    Parameters:
    - plink_prefix (str): Prefix path to the plink files (as you would use in 'plink --bfile').
    - sumstats_path (str): Path to the sumstats TSV file.
    
    Returns:
    - tuple: A tuple of Pandas DataFrames (sumstats, info_snp) ready for snp_match.

    Raises:
    - FileNotFoundError: 
    """
    bed_path = f"{plink_prefix}.bed"
    bim_path = f"{plink_prefix}.bim"
    fam_path = f"{plink_prefix}.fam"
    
    for path in [bed_path, bim_path, fam_path, sumstats_path]:
        if not os.path.exists(path):
            raise FileNotFoundError(f"File not found: {path}")
        
    G = read_plink1_bin(bed_path, bim_path, fam_path, verbose=False)
    info_snp = pd.DataFrame({
    'chr': G.chrom.values,
    'pos': G.pos.values,
    'a0': G.a0.values,
    'a1': G.a1.values,
    'rsid': G.snp.values
    })
    sumstats = pd.read_csv(sumstats_path, sep='\t')
    
    return sumstats, info_snp

def filter_standard_chromosomes(df, chr_column='chr'):
    """
    Converts the specified chromosome column to string, 
    then filters the DataFrame to keep only chromosomes 1-23.
    
    Parameters:
    - df (pd.DataFrame): Input DataFrame with a chromosome column.
    - chr_column (str): Name of the chromosome column (default is 'chr').
        
    Returns:
    - pd.DataFrame: Filtered DataFrame with only chromosomes 1-23.
    """
    df[chr_column] = df[chr_column].astype(str)
    
    df['chr_numeric'] = pd.to_numeric(df[chr_column], errors='coerce')
    
    standard_chromosomes = range(1, 24)
    
    filtered_df = df[df['chr_numeric'].isin(standard_chromosomes)].copy()
    
    filtered_df = filtered_df.drop(columns=['chr_numeric'])
    
    return filtered_df

def sigmoid(z):
    return 1 / (1 + np.exp(-z))