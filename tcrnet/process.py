import glob
import pandas as pd
import numpy as np



def standardize_tcr_data(tcr_filepath: str, 
                         technology_platform: str='10X', 
                         compression: str=None,
                         save_output: bool=False):
    """This function ingests standardizes VDJ data to allow their use with `tcrnet`"""
    valid_technology_platforms = ['10X', 'Omniscope', 'ViraFEST']
    if technology_platform == '10X':
        tcr_df = pd.read_csv(tcr_filepath, compression=compression)
        tcr_df = tcr_df.loc[(tcr_df['productive']==True)
                           &(tcr_df['is_cell']==True)].copy()
    elif technology_platform == 'Omniscope':
        raise NotImplementedError
    elif technology_platform == 'ViraFEST':
        raise NotImplementedError
    else:
        raise ValueError(f"{technology_platform} data not expected by the tool. Valid options are{valid_technology_platforms}")
    if save_output:
        tcr_df.to_csv(tcr_filepath.replace('.csv', '_standardized.csv'))
    return tcr_df


def preprocess_tcr_data(tcr_df: pd.DataFrame,
                        sample_id: str,
                        clonotype_definition: list=['cdr1', 'cdr2', 'cdr3'],
                        chain: str='alpha-beta'):
    """This function ingests VDJ data from 10X, Omniscope, or ViraFEST 
    and performs preprocessing steps, including chain-pairing if applicable."""
    # Valid options for immune repertoire receptor chains
    valid_chain_opts = ['alpha', 'beta', 'alpha-beta']
    print(f"# records before QC: {tcr_df.shape}")
    # Filter out records that are: non-productive, missing V/J calls
    for sequence_name in clonotype_definition:
        tcr_df = tcr_df.loc[~(tcr_df[sequence_name].isna())].copy()
    print(f"# records after QC 1 - Missing CDR sequences: {tcr_df.shape}")
    tcr_df['chain_identifier'] = tcr_df.apply(lambda row: '_'.join(row[col] for col in clonotype_definition), 
                                              axis=1)
    if chain=='alpha-beta':
        beta_tcr_df = tcr_df.loc[tcr_df['chain']=='TRB'].copy()
        alpha_tcr_df = tcr_df.loc[tcr_df['chain']=='TRA'].copy()
        tcr_df = pd.merge(beta_tcr_df, alpha_tcr_df, 
                          on='barcode', 
                          suffixes=('_beta', '_alpha'),
                          how='outer')
        tcr_df['chain_identifier_alpha'] = tcr_df['chain_identifier_alpha'].fillna('')
        tcr_df['chain_identifier_beta'] = tcr_df['chain_identifier_beta'].fillna('')
        print(f"# records after {chain} Pairing: {tcr_df.shape}")
        tcr_df = _apply_alpha_beta_qc(tcr_df)
        print(f"# records after QC2 - Missing Chain Pairings: {tcr_df.shape}")
    elif chain=='beta':
        tcr_df['chain_identifier_alpha'] = ''
        # TODO: implement apply_alpha_qc
    elif chain=='alpha':
        tcr_df['chain_identifier_beta'] = ''
        # TODO: implement apply_beta_qc
    else: 
        raise ValueError(f"{chain} is not a valid mode for processing alpha-beta TCRs. Choose from {valid_chain_opts}")
    tcr_df['sample_id'] = sample_id
    return tcr_df


def _apply_alpha_beta_qc(paired_tcr_df: pd.DataFrame):
    """This internal function ingests VDJ data during processing and applies quality control (QC) steps.
    This function gets automatically applied when using the `preprocess_tcr_data()` function."""
    # generate dictionaries of alpha-beta pairing information
    chain_qc_df = paired_tcr_df.loc[(paired_tcr_df['chain_identifier_alpha']!='')
                                   &(paired_tcr_df['chain_identifier_beta']!='')].copy()
    alpha_beta_seq_dict = dict(zip(chain_qc_df['chain_identifier_alpha'], chain_qc_df['chain_identifier_beta']))
    beta_alpha_seq_dict = dict(zip(chain_qc_df['chain_identifier_beta'], chain_qc_df['chain_identifier_alpha']))
    alpha_beta_umis_dict = dict(zip(chain_qc_df['chain_identifier_alpha'], chain_qc_df['umis_beta']))
    beta_alpha_umis_dict = dict(zip(chain_qc_df['chain_identifier_beta'], chain_qc_df['umis_alpha']))
    alpha_beta_reads_dict = dict(zip(chain_qc_df['chain_identifier_alpha'], chain_qc_df['reads_beta']))
    beta_alpha_reads_dict = dict(zip(chain_qc_df['chain_identifier_beta'], chain_qc_df['reads_alpha']))
    # separate out barcodes (cell IDs) with orphan alpha/beta chains
    orphan_alpha_df = paired_tcr_df.loc[(paired_tcr_df['chain_identifier_alpha']=='')].copy()
    orphan_beta_df = paired_tcr_df.loc[(paired_tcr_df['chain_identifier_beta']=='')].copy()
    # impute missing alpha pairings by using pairings from other barcodes in the dataset 
    orphan_alpha_df['chain_identifier_alpha'] = orphan_alpha_df['chain_identifier_beta'].apply(lambda x: beta_alpha_seq_dict.get(x, ''))
    orphan_alpha_df['umis_alpha'] = orphan_alpha_df['chain_identifier_beta'].apply(lambda x: beta_alpha_umis_dict.get(x, 0))
    orphan_alpha_df['reads_alpha'] = orphan_alpha_df['chain_identifier_beta'].apply(lambda x: beta_alpha_reads_dict.get(x, 0))
    # remove any remaining orphan alpha chains
    orphan_alpha_df = orphan_alpha_df.loc[orphan_alpha_df['chain_identifier_alpha']!=''].fillna('')
    # impute missing beta pairings by using pairings from other barcodes in the dataset
    orphan_beta_df['chain_identifier_beta'] = orphan_beta_df['chain_identifier_alpha'].apply(lambda x: alpha_beta_seq_dict.get(x, ''))
    orphan_beta_df['umis_beta'] = orphan_beta_df['chain_identifier_alpha'].apply(lambda x: alpha_beta_umis_dict.get(x, 0))
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
    tcr_paired_qc_df.sort_values(by=['barcode', 'umis_alpha', 
                                     'umis_beta', 'reads_alpha', 'reads_beta'], 
                                 ascending=[True, False, 
                                           False, False, False], 
                                 inplace=True)
    # filter out alpha-beta chain pairs for each cell based on higher UMI, then highest reads
    valid_clonotypes = tcr_paired_qc_df.drop_duplicates(subset=['barcode'], keep='first')['clonotype_id'].unique()
    tcr_paired_qc_df = tcr_paired_qc_df.loc[tcr_paired_qc_df['clonotype_id'].isin(valid_clonotypes)].copy()
    return tcr_paired_qc_df


def compute_clonotype_abundances(processed_tcr_df: pd.DataFrame,
                                 clonotype_definition: list=['cdr1', 'cdr2', 'cdr3'],
                                 chain: str='alpha-beta'):
    """This function computes clonotype abundances from processed VDJ data."""
    # compute number of cells per clonotype
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
            cdr_sequence_fields.extend([f'alpha_{sequence_name}'])
    if chain == 'alpha-beta' or chain == 'beta':
        for i, sequence_name in enumerate(clonotype_definition):
            tcr_counts[f'beta_{sequence_name}'] = tcr_counts['clonotype_id'].apply(lambda x: x.split('-')[1].split('_')[i])
            cdr_sequence_fields.extend([f'beta_{sequence_name}'])
    # compute sequence lengths
    for cdr_seq in cdr_sequence_fields:
        tcr_counts[f'{cdr_seq}_length'] = tcr_counts[cdr_seq].str.len()
    return tcr_counts.sort_values(by='pct_records', ascending=False)