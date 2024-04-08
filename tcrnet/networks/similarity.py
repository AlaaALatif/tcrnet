
import numpy as np
import pandas as pd
import tcrdist
import pwseqdist as pw
from tcrdist.public import _neighbors_fixed_radius



def compute_tcrdist(qtcr_df: pd.DataFrame,
                    num_cpus: int=5):
    ntcr_df = qtcr_df.rename(columns = {
    'num_records': 'count',
    'pct_records': 'frequency',
    'beta_cdr1': 'cdr1_b_aa',
    'beta_cdr2': 'cdr2_b_aa',
    'beta_cdr3': 'cdr3_b_aa',
    'alpha_cdr1': 'cdr1_a_aa',
    'alpha_cdr2': 'cdr2_a_aa',
    'alpha_cdr3': 'cdr3_a_aa',
    'clonotype_id': 'clone_id'
    }).copy()
    target_columns = ['count', 'frequency', 
                    'cdr1_b_aa','cdr2_b_aa', 'cdr3_b_aa', 
                    'cdr1_a_aa','cdr2_a_aa', 'cdr3_a_aa', 
                    'clone_id', 'sample_id']
    ntcr_df = ntcr_df[target_columns].copy()
    # assign distance metrics and weighting for each TCR sequence
    beta_metrics = {
        "cdr3_b_aa": pw.metrics.nb_vector_tcrdist,
        "cdr2_b_aa": pw.metrics.nb_vector_tcrdist,
        "cdr1_b_aa": pw.metrics.nb_vector_tcrdist,
        "cdr3_a_aa": pw.metrics.nb_vector_tcrdist,
        "cdr2_a_aa": pw.metrics.nb_vector_tcrdist,
        "cdr1_a_aa": pw.metrics.nb_vector_tcrdist
    }

    beta_weights = {
        "cdr3_b_aa": 3,
        "cdr2_b_aa": 1,
        "cdr1_b_aa": 1,
        "cdr3_a_aa": 3,
        "cdr2_a_aa": 1,
        "cdr1_a_aa": 1
    }
    beta_kargs = {
        "cdr3_b_aa": {"use_numba": True},
        "cdr2_b_aa": {"use_numba": True},
        "cdr1_b_aa": {"use_numba": True},
        "cdr3_a_aa": {"use_numba": True},
        "cdr2_a_aa": {"use_numba": True},
        "cdr1_a_aa": {"use_numba": True}
    }
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