import numpy as np
import pandas as pd
import networkx as nx
from .similarity import compute_nearest_neighbors



def generate_graph_dataframe(ntcr_df: pd.DataFrame,
                             distance_matrix: np.array,
                             clonotype_definition: list=['cdr1', 'cdr2', 'cdr3'],
                             chain: str='alpha-beta',
                             analysis_mode: str='private',
                             edge_threshold: int=64,
                             count_column_name: str='count',
                             count_threshold: int=2):
    # Valid options for immune repertoire receptor chains
    valid_chain_opts = ['alpha', 'beta', 'alpha-beta']
    # Initialize list collecting network data
    network = list()
    # create set of nearest neighbors for each clonotype
    nearest_neighbors = compute_nearest_neighbors(distance_matrix, edge_threshold)
    # collect CDR-related field names
    alpha_cdr_fields = {
        f'alpha_{sn}': f"{sn.split('_')[0]}_a_aa" if 'aa' in sn else f"{sn.split('_')[0]}_a_nt"
        for sn in clonotype_definition
    }
    beta_cdr_fields = {
        f'beta_{sn}': f"{sn.split('_')[0]}_b_aa" if 'aa' in sn else f"{sn.split('_')[0]}_b_nt"
        for sn in clonotype_definition
    }
    if chain == 'alpha-beta':
        all_field_names = list(alpha_cdr_fields.values()) + list(beta_cdr_fields.values())
    elif chain == 'beta':
        all_field_names = list(beta_cdr_fields.values())
        ntcr_df = ntcr_df.rename(columns=beta_cdr_fields).copy()
    elif chain == 'alpha':
        all_field_names = list(alpha_cdr_fields.values())
        ntcr_df = ntcr_df.rename(columns=alpha_cdr_fields).copy()
    else: 
        raise ValueError(f"{chain} is not a valid mode for processing alpha-beta TCRs. Choose from {valid_chain_opts}")
    # populate our network with nodes as clonotypes, and edges as similarity-distance between them
    for n1_idx, n1_neighbors in enumerate(nearest_neighbors):
        for n2_idx in n1_neighbors:
            record_data = []
            if n1_idx!=n2_idx:
                # record_data = []
                record_data.extend([
                    n1_idx,
                    n2_idx,
                    distance_matrix[n1_idx, n2_idx],
                ])
                for cdr_field_name in all_field_names:
                    # For each field name, select the value at n1_idx and n2_idx
                    record_data.append(ntcr_df[cdr_field_name].iloc[n1_idx])
                    record_data.append(ntcr_df[cdr_field_name].iloc[n2_idx])
                record_data.extend([
                    len(n1_neighbors),
                    False,
                    ntcr_df['clone_id'].iloc[n1_idx],
                    ntcr_df['clone_id'].iloc[n2_idx],
                    ntcr_df['sample_id'].iloc[n1_idx],
                    ntcr_df['sample_id'].iloc[n2_idx],
                    ntcr_df['count'].iloc[n1_idx],
                    ntcr_df['count'].iloc[n2_idx],
                    ntcr_df['frequency'].iloc[n1_idx],
                    ntcr_df['frequency'].iloc[n2_idx]
                ])
                network.append(record_data)
            elif ntcr_df[count_column_name].iloc[n1_idx] >= count_threshold and len(n1_neighbors)==1:
                # record_data = []
                record_data.extend([
                    n1_idx,
                    n2_idx,
                    distance_matrix[n1_idx, n2_idx]
                ])
                for cdr_field_name in all_field_names:
                    # For each field name, select the value at n1_idx and n2_idx
                    record_data.append(ntcr_df[cdr_field_name].iloc[n1_idx])
                    record_data.append(ntcr_df[cdr_field_name].iloc[n2_idx])
                record_data.extend([
                    len(n1_neighbors),
                    True,
                    ntcr_df['clone_id'].iloc[n1_idx],
                    ntcr_df['clone_id'].iloc[n2_idx],
                    ntcr_df['sample_id'].iloc[n1_idx],
                    ntcr_df['sample_id'].iloc[n2_idx],
                    ntcr_df['count'].iloc[n1_idx],
                    ntcr_df['count'].iloc[n2_idx],
                    ntcr_df['frequency'].iloc[n1_idx],
                    ntcr_df['frequency'].iloc[n2_idx]
                ])
                network.append(record_data)
    # create a dataframe representation of our network graph
    network_columns = ['node_1', 'node_2', 'distance']
    for cdr_field_name in all_field_names:
        network_columns.append(f'{cdr_field_name}_1') 
        network_columns.append(f'{cdr_field_name}_2')
    network_columns += ['k_neighbors', 'is_island',
                        'clone_id_1', 'clone_id_2',
                        'sample_id_1', 'sample_id_2',
                        'count_1', 'count_2',
                        'frequency_1', 'frequency_2']
    network_df = pd.DataFrame(network, columns = network_columns)
    # calculate the weight for each edge (connection between two TCR clonotypes)
    network_df['weight'] = (edge_threshold - network_df['distance']) / edge_threshold
    # create a field that tells us whether a connection is within a subject or between two different subjects
    network_df['relation'] = 'private'
    network_df.loc[network_df['sample_id_1']!=network_df['sample_id_2'], 'relation'] = 'public'
    if analysis_mode == 'private':
        # filter for private connections only
        network_df = network_df.loc[(network_df['relation']=='private')].copy()
    return network_df


def update_df_with_cluster_information(net_df: pd.DataFrame, partition: dict):
    cols_of_interest = ['node_1', 'node_2', 'distance',
                        'clone_id_1', 'clone_id_2','cluster_1', 'cluster_2']
    # cluster-based quantifications
    net_df['cluster_1'] = net_df['node_1'].apply(lambda x: partition.get(x, None))
    net_df['cluster_2'] = net_df['node_2'].apply(lambda x: partition.get(x, None))
    # sizes of each cluster in the left-hand side
    cluster1_sizes = (net_df
                    # .loc[net_df['cluster_1']<=top_k_clusters]
                    .groupby('cluster_1')
                    .agg(cluster1_size=('node_1', 'nunique'))
                    .reset_index())
    # sizes of each cluster in the right-hand side
    cluster2_sizes = (net_df
                    .groupby('cluster_2')
                    # .loc[net_df['cluster_2']<=top_k_clusters]
                    .agg(cluster2_size=('node_2', 'nunique'))
                    .reset_index())
    # adding cluster size information to our network dataframe
    net_df = pd.merge(net_df, cluster1_sizes, on='cluster_1', how='left')
    net_df = pd.merge(net_df, cluster2_sizes, on='cluster_2', how='left')
    net_df['cluster1_size'] = net_df['cluster1_size'].fillna(0)
    net_df['cluster2_size'] = net_df['cluster2_size'].fillna(0)
    net_df[cols_of_interest].head()
    return net_df


def create_undirected_graph(net_df: pd.DataFrame):
    # construct undirected weighted graph of TCR clonotypes
    graph = nx.from_pandas_edgelist(pd.DataFrame({
        'source': net_df.loc[net_df['is_island']==False, 'node_1'],
        'target': net_df.loc[net_df['is_island']==False, 'node_2'],
        'weight': net_df.loc[net_df['is_island']==False, 'weight'],
        'relation': net_df.loc[net_df['is_island']==False, 'relation']
        }),
        edge_attr=['weight', 'relation'])
    # add island nodes to the graph for visualization
    for i, node_island in net_df.loc[net_df['is_island']==True].iterrows():
        graph.add_node(node_island['node_1'])
    return graph


def compute_node_positions(graph,
                           k: float=0.15,
                           seed: int=42):
    node_positions = nx.spring_layout(graph, seed=42, k=.15)
    return node_positions