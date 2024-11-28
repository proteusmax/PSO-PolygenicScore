import pandas as pd
from pandas_plink import read_plink1_bin
import re
import json
import os
import numpy as np
import warnings

def map_columns(df, mapping):
    rename_dict = {}
    for target_name, possible_names in mapping.items():
        for col in df.columns:
            if col in possible_names:
                rename_dict[col] = target_name
                break  # Stop if we found a match for this target_name
    return rename_dict

def load_data_for_snp_match(sumstats_path):
    """
    Reads the info_snp and sumstats TSV files into Pandas DataFrames
    and returns them in the correct format for the snp_match function.
    
    Parameters:
    - sumstats_path (str): Path to the sumstats TSV file.
    
    Returns:
    - A Pandas DataFrames (sumstats) ready for snp_match.
    """
    sumstats = pd.read_csv(sumstats_path, sep='\t')
    sumstats.columns = [re.sub(r"\s+", "", col).lower() for col in sumstats.columns]
    
    with open("GeneticFunctions/column_mapping.json", "r") as f:
        column_mapping = json.load(f)
    rename_dict = map_columns(sumstats, column_mapping)
    sumstats.rename(columns=rename_dict, inplace=True)

    return sumstats

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
    z = np.clip(z, -500, 500)
    return 1 / (1 + np.exp(-z))
