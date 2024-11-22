
import numpy as np
import pandas as pd
import tcrdist
import pwseqdist as pw
from tcrdist.public import _neighbors_fixed_radius
from tcrdist.repertoire import TCRrep
from tcrdist.html_colors import get_html_colors


def compute_tcrdist(qtcr_df: pd.DataFrame,
                    clonotype_definition: list=['cdr1_aa', 'cdr2_aa', 'cdr3_aa'],
                    chain: str='alpha-beta',
                    clonotype_count_column: str='num_records',
                    clonotype_frequency_column: str='pct_records',
                    num_cpus: int=5):
    # Valid options for immune repertoire receptor chains
    valid_chain_opts = ['alpha', 'beta', 'alpha-beta']
    ntcr_df = qtcr_df.rename(columns = {
        clonotype_count_column: 'count',
        clonotype_frequency_column: 'frequency',
        'clonotype_id': 'clone_id'
        }).copy()
    alpha_cdr_fields = {
        f'alpha_{sn}': f"{sn.split('_')[0]}_a_aa" if 'aa' in sn else f"{sn.split('_')[0]}_a_nt"
        for sn in clonotype_definition
    }
    beta_cdr_fields = {
        f'beta_{sn}': f"{sn.split('_')[0]}_b_aa" if 'aa' in sn else f"{sn.split('_')[0]}_b_nt"
        for sn in clonotype_definition
    }
    print(beta_cdr_fields)
    if chain == 'alpha-beta':
        all_field_names = list(alpha_cdr_fields.values()) + list(beta_cdr_fields.values())
        ntcr_df = ntcr_df.rename(columns=alpha_cdr_fields).copy()
        ntcr_df = ntcr_df.rename(columns=beta_cdr_fields).copy()
    elif chain == 'beta':
        all_field_names = list(beta_cdr_fields.values())
        ntcr_df = ntcr_df.rename(columns=beta_cdr_fields).copy()
    elif chain == 'alpha':
        all_field_names = list(alpha_cdr_fields.values())
        ntcr_df = ntcr_df.rename(columns=alpha_cdr_fields).copy()
    else: 
        raise ValueError(f"{chain} is not a valid mode for processing alpha-beta TCRs. Choose from {valid_chain_opts}")
    target_columns = ['count', 'frequency']
    target_columns += all_field_names
    target_columns += ['clone_id', 'sample_id']
    ntcr_df = ntcr_df[target_columns].copy()
    # assign distance metrics and weighting for each TCR sequence
    beta_metrics = {
        cdr_field_name: pw.metrics.nb_vector_tcrdist for cdr_field_name in all_field_names
        }
    # print(f"Beta metrics: {beta_metrics}")
    beta_weights = {
        cdr_field_name: 3 if 'cdr3' in cdr_field_name else 1 for cdr_field_name in all_field_names
        }
    # print(f"Beta weights: {beta_weights}")
    beta_kargs = {
        cdr_field_name: {"use_numba": True} for cdr_field_name in all_field_names
        }
    print(f"Beta kargs: {beta_weights}")
    # compute similarity measures using tcrdist
    distance_dict = tcrdist.rep_funcs._pws(df = ntcr_df,
                        metrics = beta_metrics,
                        weights = beta_weights,
                        kargs = beta_kargs,
                        cpu = num_cpus,
                        uniquify = True,
                        store = True)
    return ntcr_df, distance_dict['tcrdist']


def compute_nearest_neighbors(distance_matrix: np.array, 
                              edge_threshold: int=150):
    # create set of nearest neighbors for each clonotype
    x = _neighbors_fixed_radius(distance_matrix, edge_threshold)
    return x