from tqdm import tqdm
import sys
import glob
import pandas as pd
import numpy as np
import scipy.stats as stats
from statsmodels.stats.multitest import fdrcorrection



def standardize_tcr_data(tcr_filepath: str, 
                         clonotype_definition: list=['cdr1_nt', 'cdr2_nt', 'cdr3_nt'],
                         technology_platform: str='10X', 
                         compression: str=None,
                         save_output: bool=False):
    """This function ingests standardizes VDJ data to allow their use with `tcrnet`"""
    valid_technology_platforms = ['10X', 'Omniscope', 'ViraFEST']
    if technology_platform.lower() == '10x':
        tcr_df = pd.read_csv(tcr_filepath, compression=compression)
        tcr_df = tcr_df.loc[(tcr_df['productive']==True)
                           &(tcr_df['is_cell']==True)].copy()
        tcr_df.rename(columns={
            # "cdr1": "cdr1_aa",
            # "cdr2": "cdr2_aa",
            # "cdr3": "cdr3_aa",
            "umis": "umi_count"
        }, inplace=True)
    elif technology_platform.lower() == 'omniscope':
        # Load TCR file
        tcr_df = pd.read_csv(tcr_filepath, compression=compression)
        # Filter for TCR records that are productive
        tcr_df = tcr_df.loc[(tcr_df['productive']=='T')].copy()
        # Column name reformatting
        tcr_df.rename(columns={
            "cdr1": "cdr1_nt",
            "cdr2": "cdr2_nt",
            "cdr3": "cdr3_nt",
            "locus": "chain",
        }, inplace=True)
        tcr_df.reset_index(inplace=True, drop=True)
        # Create pseudo-barcode for each TCR record.
        tcr_df['barcode'] = tcr_df.index
        # Reformat VDJ gene calls -- remove allelic information.
        tcr_df['v_call'] = tcr_df['v_call'].apply(lambda x: x.split("*")[0])
        tcr_df['v_call'] = tcr_df['v_call'].apply(lambda x: x.split("*")[0])
        tcr_df['j_call'] = tcr_df['j_call'].apply(lambda x: x.split("*")[0])
    elif technology_platform.lower() == 'virafest':
        raise NotImplementedError
    else:
        raise ValueError(f"{technology_platform} data not expected by the tool. Valid options are{valid_technology_platforms}")
    # Replace missing sequence values with empty strings
    for sequence_name in clonotype_definition:
        tcr_df[sequence_name] = tcr_df[sequence_name].fillna('')
        tcr_df[sequence_name] = tcr_df[sequence_name].astype(str)
    if save_output:
        tcr_df.to_csv(tcr_filepath.replace('.csv', '_standardized.csv'))
    return tcr_df


def preprocess_tcr_data(tcr_df: pd.DataFrame,
                        clonotype_definition: list=['cdr1', 'cdr2', 'cdr3'],
                        chain: str='alpha-beta',
                        sample_id: str=""):
    """This function ingests VDJ data from 10X, Omniscope, or ViraFEST 
    and performs preprocessing steps, including chain-pairing if applicable."""
    # Valid options for immune repertoire receptor chains
    valid_chain_opts = ['alpha', 'beta', 'alpha-beta']
    print(f"# records before QC: {tcr_df.shape}")
    # Filter out records that are missing sequence information
    for sequence_name in clonotype_definition:
        tcr_df = tcr_df.loc[(~tcr_df[sequence_name].isna())
                           &(tcr_df[sequence_name]!='')].copy()
    print(f"# records after QC 1 - Missing CDR sequences: {tcr_df.shape}")
    tcr_df['chain_identifier'] = tcr_df.apply(lambda row: '_'.join(row[col] for col in clonotype_definition), 
                                              axis=1)
    if chain=='alpha-beta':
        # separate out beta chains
        beta_tcr_df = tcr_df.loc[tcr_df['chain']=='TRB'].copy()
        # separate out alpha chains
        alpha_tcr_df = tcr_df.loc[tcr_df['chain']=='TRA'].copy()
        # perform outer merge between alpha and beta chain data
        tcr_df = pd.merge(beta_tcr_df, alpha_tcr_df, 
                          on='barcode', 
                          suffixes=('_beta', '_alpha'),
                          how='outer')
        tcr_df['chain_identifier_alpha'] = tcr_df['chain_identifier_alpha'].fillna('')
        tcr_df['chain_identifier_beta'] = tcr_df['chain_identifier_beta'].fillna('')
        print(f"# records after {chain} Pairing: {tcr_df.shape}")
        tcr_df = _apply_alpha_beta_qc(tcr_df)
        print(f"# records after QC 2 - Missing Chain Pairings: {tcr_df.shape}")
    elif chain=='beta':
        tcr_df.rename(columns={"chain_identifier": "chain_identifier_beta"}, inplace=True)
        tcr_df['chain_identifier_alpha'] = ''
        tcr_df = _apply_beta_qc(tcr_df)
        print(f"# records after QC 2 - Missing Beta-Chain Sequence Information: {tcr_df.shape}")
    elif chain=='alpha':
        tcr_df.rename(columns={"chain_identifier": "chain_identifier_alpha"}, inplace=True)
        tcr_df['chain_identifier_beta'] = ''
        tcr_df = _apply_alpha_qc(tcr_df)
        print(f"# records after QC 2 - Missing Alpha-Chain Sequence Information: {tcr_df.shape}")
    else: 
        raise ValueError(f"{chain} is not a valid mode for processing alpha-beta TCRs. Choose from {valid_chain_opts}")
    if "sample_id" not in tcr_df.columns:
        tcr_df['sample_id'] = sample_id
    return tcr_df


def downsample_tcr_dataframes(tcrdfs_list: list):
    """This function downsamples each dataframe in the input list based on the minimum number of 
    TCR records in the list provided."""
    # compute downsampling size
    downsample_size = sys.maxsize
    for vdjdf in tcrdfs_list:
        downsample_size = min(vdjdf.shape[0], downsample_size)
    # downsample each vdjdf
    downsampled_tcrdfs_list = []
    for tcrdf in tcrdfs_list:
        downsampled_tcrdfs_list.append(tcrdf.sample(downsample_size))
    return downsampled_tcrdfs_list


def compute_clonotype_abundances(processed_tcr_df: pd.DataFrame,
                                 clonotype_definition: list=['cdr1_aa', 'cdr2_aa', 'cdr3_aa'],
                                 abundance_column_name: str=None,
                                 chain: str='alpha-beta'):
    """This function computes clonotype abundances from processed VDJ data."""
    # Valid options for immune repertoire receptor chains
    valid_chain_opts = ['alpha', 'beta', 'alpha-beta']
    # compute number of cells per clonotype
    if abundance_column_name:
        tcr_counts = (processed_tcr_df
                  .groupby(['sample_id', 'clonotype_id'])
                  .agg(num_records=(abundance_column_name, 'sum'))
                  .reset_index())
    else:
        tcr_counts = (processed_tcr_df
                  .groupby(['sample_id', 'clonotype_id'])
                  .agg(num_records=('barcode', 'nunique'))
                  .reset_index())
    tcr_total_counts = (tcr_counts
                  .groupby('sample_id')
                  .agg(total_records=('num_records', 'sum'))
                  .reset_index())
    tcr_counts = pd.merge(tcr_counts, tcr_total_counts, on='sample_id')
    # compute relative frequency of each clonotype
    tcr_counts['pct_records'] = tcr_counts['num_records'] / tcr_counts['total_records']
    # extract CDR sequences
    cdr_sequence_fields = []
    if chain == 'alpha-beta' or chain == 'alpha':
        for i, sequence_name in enumerate(clonotype_definition):
            tcr_counts[f'alpha_{sequence_name}'] = tcr_counts['clonotype_id'].apply(lambda x: x.split('-')[0].split('_')[i])
            tcr_counts[f'beta_{sequence_name}'] = tcr_counts['clonotype_id'].apply(lambda x: x.split('-')[1].split('_')[i])
            cdr_sequence_fields.extend([f'alpha_{sequence_name}', 
                                        f'beta_{sequence_name}'])
    elif chain == 'beta':
        for i, sequence_name in enumerate(clonotype_definition):
            tcr_counts[f'beta_{sequence_name}'] = tcr_counts['clonotype_id'].apply(lambda x: x.split('_')[i])
            cdr_sequence_fields.extend([f'beta_{sequence_name}'])
    elif chain == 'alpha':
        for i, sequence_name in enumerate(clonotype_definition):
            tcr_counts[f'alpha_{sequence_name}'] = tcr_counts['clonotype_id'].apply(lambda x: x.split('_')[i])
            cdr_sequence_fields.extend([f'alpha_{sequence_name}'])
    else:
        raise ValueError(f"{chain} is not a valid chain type. Choose from {valid_chain_opts}")
    # compute sequence lengths
    for cdr_seq in cdr_sequence_fields:
        tcr_counts[f'{cdr_seq}_length'] = tcr_counts[cdr_seq].str.len()
    return tcr_counts.sort_values(by='pct_records', ascending=False)

def compute_longitudinal_clonotype_expansion(vdj_counts_basket: tuple,
                                             foldchange_threshold: float,
                                             pvalue_threshold: float,
                                             clonotype_definition: list,
                                             chain: str):
    """This function fuses clonotype data from PRE/POST treatment of the same PID."""
    if len(vdj_counts_basket)==1:
        raise ValueError(f"Input data basket has 1 dataframe, we expect atleast 2 dataframes to perform longitudinal analyses")
    elif len(vdj_counts_basket)>2:
        raise NotImplementedError(f"Input data basket has more than 2 dataframes, this function currently only supports pair-wise analyses")
    vdj_mrg_counts = pd.merge(vdj_counts_basket[0], vdj_counts_basket[1], 
                              on='clonotype_id', how='outer', suffixes=('_pre', '_post'))
    # Add missing values for missing clonotypes (i.e. 0s)
    vdj_mrg_counts = vdj_mrg_counts.fillna(0)
    # Consolidate sample ID
    vdj_mrg_counts['sample_id'] = vdj_mrg_counts['sample_id_post']
    # Consolidate sequence information
    for sn in clonotype_definition:
        if 'alpha' in chain:
            vdj_mrg_counts[f'alpha_{sn}'] = vdj_mrg_counts[f'alpha_{sn}_pre']
            vdj_mrg_counts.loc[vdj_mrg_counts[f'alpha_{sn}_pre']==0, f'alpha_{sn}'] = vdj_mrg_counts[f'alpha_{sn}_post']
        if 'beta' in chain:
            vdj_mrg_counts[f'beta_{sn}'] = vdj_mrg_counts[f'beta_{sn}_pre']
            vdj_mrg_counts.loc[vdj_mrg_counts[f'beta_{sn}_pre']==0, f'beta_{sn}'] = vdj_mrg_counts[f'beta_{sn}_post']
    vdj_mrg_counts.loc[vdj_mrg_counts['sample_id_post']==0, 'sample_id'] = vdj_mrg_counts['sample_id_pre']
    vdj_mrg_counts['sample_id'] = vdj_mrg_counts['sample_id'].apply(lambda x: x.split('_')[0])
    vdj_mrg_counts['num_records_diff'] = vdj_mrg_counts['num_records_post'] - vdj_mrg_counts['num_records_pre']
    vdj_mrg_counts['pct_records_diff'] = vdj_mrg_counts['pct_records_post'] - vdj_mrg_counts['pct_records_pre']
    num_expanded_clonotypes = vdj_mrg_counts.loc[vdj_mrg_counts['num_records_diff']>0].shape
    print(f"# Clonotypes that expanded post treatment: {num_expanded_clonotypes}")
    mean_expansion = vdj_mrg_counts.loc[vdj_mrg_counts['num_records_diff']>0, 'pct_records_diff'].mean()
    print(f"Mean proportional clonotype expansion post treatment: {mean_expansion*100}%")
    num_contracted_clonotypes = vdj_mrg_counts.loc[vdj_mrg_counts['num_records_diff']<0].shape
    print(f"# Clonotypes that contracted post treatment: {num_contracted_clonotypes}")
    mean_contraction = -1*vdj_mrg_counts.loc[vdj_mrg_counts['num_records_diff']<0, 'pct_records_diff'].mean()
    print(f"Mean proportional clonotype contraction post treatment: {mean_contraction*100}%")
    # add 1 to each clonotype count i.e. adding pseudocounts to clonotypes missing in one of the two timepoints
    vdj_mrg_counts['num_records_pre'] += 1
    vdj_mrg_counts['num_records_post'] += 1
    # compute total counts pre/post vaccine for the patient
    vdj_mrg_counts['total_pre'] = vdj_mrg_counts['num_records_pre'].sum()
    vdj_mrg_counts['total_post'] = vdj_mrg_counts['num_records_post'].sum()
    # compute relative clonotype frequency
    vdj_mrg_counts['frequency_records_pre'] = vdj_mrg_counts['num_records_pre'] / vdj_mrg_counts['total_pre']
    vdj_mrg_counts['frequency_records_post'] = vdj_mrg_counts['num_records_post'] / vdj_mrg_counts['total_post']
    # compute log10 transformation of relative frequencies
    vdj_mrg_counts['log10_freq_pre'] = np.log10(vdj_mrg_counts['frequency_records_pre'])
    vdj_mrg_counts['log10_freq_post'] = np.log10(vdj_mrg_counts['frequency_records_post'])
    # compute log2 fold-change in clonotype frequency after vaccination
    vdj_mrg_counts['log2FC'] = np.log2(vdj_mrg_counts['frequency_records_post']/vdj_mrg_counts['frequency_records_pre'])
    # compute p-values from Fisher's exact test on absolute counts
    vdj_mrg_counts['fisher_pvalue'] = apply_fishers_test(vdj_mrg_counts)
    # vdj_mrg_counts['fisher_pvalue'] = vdj_mrg_counts.apply(apply_fishers_test, axis=1)
    # compute adjusted p-values using false discovery rate
    print("p-value correction for multiple hypothesis testing.")
    vdj_mrg_counts['fisher_adjusted_pvalue'] = fdrcorrection(vdj_mrg_counts['fisher_pvalue'])[1]
    # categorize each clonotype as either stable, expanded, or contracted
    vdj_mrg_counts['clonotype_dynamic_state'] = 'stable'
    vdj_mrg_counts.loc[(vdj_mrg_counts['log2FC']>=foldchange_threshold)
                   &(vdj_mrg_counts['fisher_adjusted_pvalue']<pvalue_threshold), 
                    'clonotype_dynamic_state'] = 'expanded'
    vdj_mrg_counts.loc[(vdj_mrg_counts['log2FC']<=-foldchange_threshold)
                   &(vdj_mrg_counts['fisher_adjusted_pvalue']<pvalue_threshold), 
                    'clonotype_dynamic_state'] = 'contracted'
    # reverse the effects of the pseudocounts before returning final results
    vdj_mrg_counts['num_records_pre'] -= 1
    vdj_mrg_counts['num_records_post'] -= 1
    # compute total counts pre/post vaccine for the patient
    vdj_mrg_counts['total_pre'] = vdj_mrg_counts['num_records_pre'].sum()
    vdj_mrg_counts['total_post'] = vdj_mrg_counts['num_records_post'].sum()
    # classify each clonotype as either newly or dual expanded/contracted
    vdj_mrg_counts['clonotype_dynamic_mode'] = 'newly'
    vdj_mrg_counts.loc[(vdj_mrg_counts['clonotype_dynamic_state']=='expanded')
                      &(vdj_mrg_counts['num_records_pre']>=1), 'clonotype_dynamic_mode'] = 'dual'
    vdj_mrg_counts.loc[(vdj_mrg_counts['clonotype_dynamic_state']=='contracted')
                      &(vdj_mrg_counts['num_records_post']>=1), 'clonotype_dynamic_mode'] = 'dual'
    return vdj_mrg_counts.sort_values(by=['log2FC', 'num_records_diff'], ascending=False)

def apply_fishers_test(merged_tcr_counts: pd.DataFrame):
    # Extract constants and arrays
    total_pre = merged_tcr_counts['total_pre'].iloc[0]
    total_post = merged_tcr_counts['total_post'].iloc[0]
    print("test fishers enhancement")
    pre_counts = merged_tcr_counts['num_records_pre'].values
    post_counts = merged_tcr_counts['num_records_post'].values

    # Compute p-values in a loop with a progress bar
    pvalues = []
    for pre_val, post_val in tqdm(zip(pre_counts, post_counts), total=len(pre_counts), desc="Computing Fisher's exact tests"):
        table = [[pre_val, total_pre - pre_val],
                 [post_val, total_post - post_val]]
        _, pval = stats.fisher_exact(table)
        pvalues.append(pval)

    return pvalues

    # data = [[x['num_records_pre'], x['num_records_post']], 
    #         [x['total_pre']-x['num_records_pre'], 
    #          x['total_post']-x['num_records_post']]]
    # _, pvalue = stats.fisher_exact(data)
    # return pvalue

def _apply_beta_qc(tcr_df: pd.DataFrame):
    """This internal function ingests VDJ data during processing and applies quality control (QC) steps.
    This function gets automatically applied when using the `preprocess_tcr_data()` function 
    with parameter `chain='beta'`."""
    chain_qc_df = tcr_df.loc[(tcr_df['chain_identifier_alpha']=='')
                            &(tcr_df['chain_identifier_beta']!='')].copy()
    chain_qc_df['clonotype_id'] = tcr_df['chain_identifier_beta'].copy()
    return chain_qc_df


def _apply_alpha_qc(tcr_df: pd.DataFrame):
    """This internal function ingests VDJ data during processing and applies quality control (QC) steps.
    This function gets automatically applied when using the `preprocess_tcr_data()` function 
    with parameter `chain='alpha'`."""
    chain_qc_df = tcr_df.loc[(tcr_df['chain_identifier_alpha']!='')
                            &(tcr_df['chain_identifier_beta']=='')].copy()
    chain_qc_df['clonotype_id'] = tcr_df['chain_identifier_alpha'].copy()
    return chain_qc_df


def _apply_alpha_beta_qc(paired_tcr_df: pd.DataFrame):
    """This internal function ingests VDJ data during processing and applies quality control (QC) steps.
    This function gets automatically applied when using the `preprocess_tcr_data()` function 
    with parameter `chain='alpha-beta'`."""
    # generate dictionaries of alpha-beta pairing information
    chain_qc_df = paired_tcr_df.loc[(paired_tcr_df['chain_identifier_alpha']!='')
                                   &(paired_tcr_df['chain_identifier_beta']!='')].copy()
    alpha_beta_seq_dict = dict(zip(chain_qc_df['chain_identifier_alpha'], chain_qc_df['chain_identifier_beta']))
    beta_alpha_seq_dict = dict(zip(chain_qc_df['chain_identifier_beta'], chain_qc_df['chain_identifier_alpha']))
    alpha_beta_umis_dict = dict(zip(chain_qc_df['chain_identifier_alpha'], chain_qc_df['umi_count_beta']))
    beta_alpha_umis_dict = dict(zip(chain_qc_df['chain_identifier_beta'], chain_qc_df['umi_count_alpha']))
    alpha_beta_reads_dict = dict(zip(chain_qc_df['chain_identifier_alpha'], chain_qc_df['reads_beta']))
    beta_alpha_reads_dict = dict(zip(chain_qc_df['chain_identifier_beta'], chain_qc_df['reads_alpha']))
    # separate out barcodes (cell IDs) with orphan alpha/beta chains
    orphan_alpha_df = paired_tcr_df.loc[(paired_tcr_df['chain_identifier_alpha']=='')].copy()
    orphan_beta_df = paired_tcr_df.loc[(paired_tcr_df['chain_identifier_beta']=='')].copy()
    # impute missing alpha pairings by using pairings from other barcodes in the dataset 
    orphan_alpha_df['chain_identifier_alpha'] = orphan_alpha_df['chain_identifier_beta'].apply(lambda x: beta_alpha_seq_dict.get(x, ''))
    orphan_alpha_df['umi_count_alpha'] = orphan_alpha_df['chain_identifier_beta'].apply(lambda x: beta_alpha_umis_dict.get(x, 0))
    orphan_alpha_df['reads_alpha'] = orphan_alpha_df['chain_identifier_beta'].apply(lambda x: beta_alpha_reads_dict.get(x, 0))
    # remove any remaining orphan alpha chains
    orphan_alpha_df = orphan_alpha_df.loc[orphan_alpha_df['chain_identifier_alpha']!=''].fillna('')
    # impute missing beta pairings by using pairings from other barcodes in the dataset
    orphan_beta_df['chain_identifier_beta'] = orphan_beta_df['chain_identifier_alpha'].apply(lambda x: alpha_beta_seq_dict.get(x, ''))
    orphan_beta_df['umi_count_beta'] = orphan_beta_df['chain_identifier_alpha'].apply(lambda x: alpha_beta_umis_dict.get(x, 0))
    orphan_beta_df['reads_beta'] = orphan_beta_df['chain_identifier_alpha'].apply(lambda x: alpha_beta_reads_dict.get(x, 0))
    # remove any remaining orphan beta chains
    orphan_beta_df = orphan_beta_df.loc[orphan_beta_df['chain_identifier_beta']!=''].fillna('')
    # aggregate original paired chains with imputed paired chains
    tcr_paired_qc_df = pd.concat([chain_qc_df,
                                  orphan_alpha_df, 
                                  orphan_beta_df], axis=0)
    # create identifier for beta chains
    tcr_paired_qc_df['clonotype_id'] = tcr_paired_qc_df['chain_identifier_alpha'] + '-' + tcr_paired_qc_df['chain_identifier_beta']
    # sort combined data in terms of cell -> UMIs (alpha+beta) -> reads (alpha+beta)
    tcr_paired_qc_df.sort_values(by=['barcode', 'umi_count_alpha', 
                                     'umi_count_beta', 'reads_alpha', 'reads_beta'], 
                                 ascending=[True, False, 
                                           False, False, False], 
                                 inplace=True)
    # filter out alpha-beta chain pairs for each cell based on higher UMI, then highest reads
    valid_clonotypes = tcr_paired_qc_df.drop_duplicates(subset=['barcode'], keep='first')['clonotype_id'].unique()
    tcr_paired_qc_df = tcr_paired_qc_df.loc[tcr_paired_qc_df['clonotype_id'].isin(valid_clonotypes)].copy()
    return tcr_paired_qc_df