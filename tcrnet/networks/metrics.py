import numpy as np
import pandas as pd
import networkx as nx
from itertools import combinations



def compute_network_metrics(net_df: pd.DataFrame,
                            graph: nx.classes.graph.Graph,
                            top_k_clusters: int=9):
    network_metrics = {}
    network_metrics['total_density'] = compute_network_density(net_df=net_df,
                                                               graph=graph)
    network_metrics['intracluster_density'] = compute_intracluster_density(net_df=net_df,
                                                                            top_k_clusters=top_k_clusters)
    network_metrics['intercluster_density'] = compute_intercluster_density(net_df=net_df,
                                                                           top_k_clusters=top_k_clusters)
    return network_metrics


def compute_network_density(net_df: pd.DataFrame,
                            graph: nx.classes.graph.Graph):
    # compute number of nodes in graph
    n = len(graph.nodes)
    # compute overall network density
    network_density = net_df.loc[net_df['node_1']!=net_df['node_2'], 
                                        'weight'].sum() / (n*(n-1))
    return network_density


def compute_intercluster_density(net_df: pd.DataFrame,
                                 top_k_clusters: int=9):
    clusters_of_interest = list(range(0, top_k_clusters))
    intercon_df = (net_df
                   .loc[(net_df['cluster_1'].isin(clusters_of_interest))
                       &(net_df['cluster_2'].isin(clusters_of_interest))
                       &(net_df['cluster_1']!=net_df['cluster_2'])]
                   .copy())
    intercluster_connectivity = []
    for cluster_combination in combinations(clusters_of_interest, 2):
        bicluster_df = (intercon_df
                         .loc[(intercon_df['cluster_1']==cluster_combination[0])
                             &(intercon_df['cluster_2']==cluster_combination[1])]
                         .copy())
        inter_numerator = bicluster_df['weight'].sum()**2
        inter_denominator = (bicluster_df['cluster1_size'].mean()*bicluster_df['cluster2_size'].mean()*np.abs(bicluster_df['cluster2_size'].mean() - bicluster_df['cluster1_size'].mean())**2)
        intercluster_connectivity.append(np.sqrt(inter_numerator / np.maximum(1., inter_denominator)))
    # handle NaN entries
    intercluster_connectivity = np.nan_to_num(intercluster_connectivity)
    return intercluster_connectivity.mean()


def compute_intracluster_density(net_df: pd.DataFrame,
                                 top_k_clusters: int=9):
    clusters_of_interest = list(range(0, top_k_clusters))
    intracluster_connectivity = []
    for cluster_number in clusters_of_interest:
        unicluster_df = (net_df
                            .loc[(net_df['cluster_1']==cluster_number)
                                &(net_df['cluster_2']==cluster_number)
                                &(net_df['node_1']!=net_df['node_2'])]
                            .copy())
        # compute numerator of density formula
        intra_numerator = unicluster_df['weight'].sum()
        # sanity check for intra-cluster formula
        assert np.nan_to_num(unicluster_df['cluster1_size'].mean())==np.nan_to_num(unicluster_df['cluster2_size'].mean()), "Error: unexpected non-equality"
        # compute denominator of density formula
        intra_denominator = np.maximum(1, unicluster_df['cluster1_size'].mean()*(unicluster_df['cluster2_size'].mean()-1))
        intracluster_connectivity.append(intra_numerator / intra_denominator)
    # handle NaN entries
    intracluster_connectivity = np.nan_to_num(intracluster_connectivity)
    return intracluster_connectivity.mean()