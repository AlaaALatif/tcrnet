
import numpy as np
import pandas as pd
import tcrdist
import pwseqdist as pw
from tcrdist.public import _neighbors_fixed_radius



def compute_tcrdist(qtcr_df: pd.DataFrame,
                    clonotype_definition: list=['cdr1', 'cdr2', 'cdr3'],
                    num_cpus: int=5):
    ntcr_df = qtcr_df.rename(columns = {
        'num_records': 'count',
        'pct_records': 'frequency',
        'clonotype_id': 'clone_id'
        }).copy()
    alpha_cdr_fields = {
        f'alpha_{sequence_name}': f'{sequence_name}_a_aa' for sequence_name in clonotype_definition
        }
    ntcr_df = ntcr_df.rename(columns=alpha_cdr_fields).copy()
    beta_cdr_fields = {
        f'beta_{sequence_name}': f'{sequence_name}_b_aa' for sequence_name in clonotype_definition
        }
    all_field_names = list(alpha_cdr_fields.values()) + list(beta_cdr_fields.values())
    ntcr_df = ntcr_df.rename(columns=beta_cdr_fields).copy()
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
    # print(f"Beta kargs: {beta_kargs}")
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