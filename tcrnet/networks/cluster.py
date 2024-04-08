import pandas as pd
import community as community_louvain
from tcrdist.html_colors import get_html_colors
from .graph import create_undirected_graph

def cluster_lovain(net_df: pd.DataFrame,
                   color_top_k_clusters: int=9,
                   random_state: int=42):
    # create a undirected graph
    graph = create_undirected_graph(net_df=net_df)
    # perform unsupervised clustering on the graph
    partition = community_louvain.best_partition(graph, random_state=random_state)
    partitions_by_cluster_size = list(pd.Series(partition.values()).value_counts().index)
    # order clusters based on their size
    partition_reorder = {idx: rank for idx, rank in zip(partitions_by_cluster_size, 
                                                        range(len(partitions_by_cluster_size)))}
    partition = {k: partition_reorder.get(v) for k, v in partition.items()}
    clusters = [i for i in pd.Series(partition.values()).value_counts().index[:color_top_k_clusters]]
    # assign a color for each of the top K clusters
    colors = get_html_colors(color_top_k_clusters)
    cluster2color = {clust: color for clust, color in zip(clusters, colors)}
    return partition, cluster2color