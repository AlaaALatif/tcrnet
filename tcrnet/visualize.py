import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import seaborn as sns
import networkx as nx
from .networks.graph import compute_node_positions
from .networks.metrics import compute_network_metrics



def generate_network_plot(graph: nx.classes.graph.Graph,
                          graph_df: pd.DataFrame,
                          network_metrics: dict,
                          partition: dict,
                          node_colors: dict,
                          edge_colors: dict=None,
                          node_sizes_field: str=None,
                          node_sizes_label_field: str=None,
                          node_shapes_field: str=None,
                          node_shapes: dict=None,
                          output_filepath: str=None,
                          k: float=.15,
                          seed: int=314,
                          plot_title: str='',
                          figsize: tuple=(12, 10),
                          dpi: int=500,
                          nx_kwargs: dict=None):
    # Set the desired figure resolution
    plt.rcParams['figure.dpi'] = dpi
    if not nx_kwargs:
        nx_kwargs = {"edgecolors": "none"}
    # Set the desired figure size (adjust width and height as needed)
    fig = plt.figure(figsize=figsize)
    node_positions = compute_node_positions(graph, k=k, seed=seed)
    # Generate a list of colors for the edges
    if edge_colors:
        edge_colors_list = [edge_colors[graph[u][v]['relation']] for u, v in graph.edges()]
    else:
        edge_colors_list = ['black' for u, v in graph.edges()]
    # Generate a dictionary mapping node IDs to their size representations, by node size
    if node_sizes_field:
        node_sizes_dict = dict(zip(graph_df['node_1'], graph_df[node_sizes_field]))
    else:
        graph_df['node_1_size'] = 50
        node_sizes_dict = dict(zip(graph_df['node_1'], graph_df['node_1_size']))
    # Generate a dicitionary mapping node IDs to disease states, by node shape
    node_states_dict = {}
    if node_shapes and node_shapes_field:
        for attribute_value in node_shapes.keys():
            node_states_dict[attribute_value] = graph_df.loc[graph_df[node_shapes_field]==attribute_value, 
                                                            'node_1'].unique().tolist()
    else:
        node_shapes = {'all': 'o'}
        node_states_dict['all'] = graph_df['node_1'].unique().tolist()
    # Draw TCR network nodes
    for disease_state, nodes_list in node_states_dict.items():
        # Create lists of nodes in the graph
        node_sizes_list = [node_sizes_dict[node] for node in nodes_list]
        # Draw the network graphs with circles for the 1st state and triangles for the 2nd state
        nx.draw_networkx_nodes(graph,
                nodelist=nodes_list,  # Only nodes in the 1st state
                pos=node_positions,
                node_color=[node_colors.get(partition.get(i), 'grey') for i in nodes_list],
                node_shape=node_shapes[disease_state],  # Circle shape for 1st state
                node_size=node_sizes_list,
                **nx_kwargs)
    # Draw TCR network edges (weighted)
    nx.draw_networkx_edges(graph, pos=node_positions, edge_color=edge_colors_list)
    # Annotate each cluster with its cluster number
    for cluster, color in list(node_colors.items()):
        cluster_nodes = [node for node, part in partition.items() if part == cluster]
        x, y = zip(*[node_positions[node] for node in cluster_nodes])
        x_center, y_center = sum(x) / len(cluster_nodes), sum(y) / len(cluster_nodes)
        plt.text(x_center, y_center, f'{cluster}', fontsize=11, 
                 color=color, ha='center', va='center', fontweight='bold')
    # Generate legends for node shapes
    shape_legends = []
    for disease_state, node_shape in node_shapes.items():
        shape_legends += [
            Line2D([0], [0], marker=node_shape, color='w', markerfacecolor='w', 
                   markersize=10, label=disease_state, markeredgewidth=0.5, 
                   markeredgecolor='k')
        ]
    # Generate legends for node sizes
    size_legends = []
    if node_sizes_label_field:
        node_sizes_labels = dict(zip(graph_df[node_sizes_label_field], graph_df[node_sizes_field]))
        node_sizes_labels_sorted = dict(sorted(node_sizes_labels.items(), key=lambda item: item[1]))
        for node_size_label, node_size in node_sizes_labels_sorted.items():
            size_legends += [
                Line2D([0], [0], marker='o', color='w', markerfacecolor='grey',
                       label=node_size_label, markersize=np.sqrt(node_size), markeredgecolor='w')
            ]
    # Legend customization
    first_legend = plt.legend(handles=shape_legends, loc='upper right', title='Disease State')
    ax = plt.gca()
    ax.add_artist(first_legend)
    plt.legend(handles=size_legends, loc='upper left', title='Abundance')
    # Generate title for plot
    latex_symbol1 = r'$D_{total}$'
    latex_symbol2 = r'$\overline{S}(C_{x})_{x \in [0:k]}$'
    latex_symbol3 = r'$\overline{S}(C_{x}, C_{y})_{x,y \in [0:k], x \neq y}$'
    latex_symbol4 = r'$\beta$'
    plt.title(f"""{plot_title}\n
              {latex_symbol1} = {network_metrics['total_density']:.6f}\n
              {latex_symbol2} = {np.mean(network_metrics['intracluster_density']):.4G}\n
              {latex_symbol3} = {np.mean(network_metrics['intercluster_density']):.4G}""")
    # Save network plot to user-specified filepath (optional)
    if output_filepath:
        plt.savefig(output_filepath, 
                    bbox_inches='tight',
                    dpi=dpi)
    return plt.show()


def generate_abundances_box_plot(meta: pd.DataFrame, 
                                 treatment_group: str,
                                 treatment_conditions: tuple,
                                 target_variable: str, 
                                 results_filepath: str=None,
                                 title: str="",
                                 shape_of_points: str='o',
                                 alpha: float=0.6,
                                 color_field: str=None,
                                 color_dict: dict=None,
                                 figsize=(14, 8),
                                 y_limits=None,
                                 show_count_annotations: bool=False):
    count_data = []
    data = []
    color_labels = []
    for condition in treatment_conditions:
        data.append(meta.loc[meta[treatment_group]==condition, 
                             target_variable].tolist())
        count_data.append(meta.loc[meta[treatment_group]==condition, 
                             'num_records'].tolist())
    if color_field and color_dict:
        for condition in treatment_conditions:
            color_labels.append(meta.loc[meta[treatment_group]==condition, 
                                color_field].unique()[0])
    else:
        color_dict = {
            'all': 'black'
        }
        color_labels = ['all' for condition in treatment_conditions]
    # Create a figure and axis
    fig, ax = plt.subplots(figsize=figsize)
    # customize median lines
    medianprops = {'color': 'red', 'linestyle': 'dashed'}
    # Create the box plot
    boxplot = ax.boxplot(data, patch_artist=True, showfliers=False, medianprops=medianprops)
    n_total_symbol = r'$N_{total}$'
    n_clonos_symbol = r'$N_{clonotypes}$'
    for i, data_subset in enumerate(data):
        # Add individual data points
        boxplot['boxes'][i].set(facecolor='white')  # Set the box color
        xi_jitter = np.random.normal(i+1, 0.04, size=len(data_subset))
        ax.plot(xi_jitter, data_subset, shape_of_points, alpha=alpha, 
                color=color_dict[color_labels[i]])
        if show_count_annotations:
            ax.text(i+1, np.max(data_subset)+0.2, f'{n_clonos_symbol}={len(data_subset)}', 
                    fontsize=10, ha='center', color='black')
            ax.text(i+1, np.max(data_subset)+0.25, f'{n_total_symbol}={np.sum(count_data[i])}', 
                    fontsize=10, ha='center', color='black')
    color_legends = []
    for i, label in enumerate(color_dict.keys()):
        color_legends += [
            Line2D([0], [0], marker='o', color=color_dict[label],
                    markerfacecolor=color_dict[label], markersize=10, 
                    label=label, markeredgewidth=0.5, markeredgecolor='k')
                    ]
    # Customize the plot (optional)
    plt.legend(handles=color_legends, loc='upper right', title=color_field)
    if y_limits:
        ax.set_ylim(y_limits)
    else:
        ax.set_ylim((meta[target_variable].min(), meta[target_variable].max()+0.3))
    ax.set_xlabel(f"{treatment_group}")
    ax.set_xticklabels([str(condition) for condition in treatment_conditions])  # Add labels to the groups
    ax.set_xticklabels([])
    ax.set_ylabel(f"{target_variable}")  # Add a label to the y-axis

    # Show the plot
    plt.title(f'{title}')
    plt.grid(True)  # Add grid lines (optional)
    # Save the plot
    if results_filepath:
        # Save the plot
        plt.savefig(results_filepath, bbox_inches="tight", dpi=350)
    plt.show()


def top_n_clonotypes(tcr_df: pd.DataFrame, 
                     top_n: int=13, 
                     output_filepath: str=None,
                     figsize: tuple=(8,6),
                     dpi: int=350):
    """This function generates a stacked bar plot showing the top N most abundant clonotype 
    in the processed TCR data"""
    # Keep the top N clonotypes by relative abundance from the data
    top_clonotypes = tcr_df.nlargest(top_n, columns='pct_records').reset_index()[::-1]
    # Colors for each segment (clonotype)
    colors = plt.cm.tab20(range(top_n))
    # Generate stacked bar plot
    plt.figure(figsize=figsize)
    bottom = 0  # Start the first segment at the base
    for i, row in top_clonotypes.iterrows():
        plt.bar(' ', row['pct_records']*100, bottom=bottom, 
                color=colors[i], label=row['clonotype_id'])
        plt.text(' ', bottom + row['pct_records']*100/2, 
                 f"{row['clonotype_id']} ({row['pct_records']*100:.3f}%, {row['num_records']})", 
                 ha='center', va='center')
        bottom += row['pct_records']*100  # Move the starting point of the next segment up
    # Remove top and right borders
    ax = plt.gca()  # Get current axes
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.xlabel(f'Top {top_n} Clonotypes', fontsize=16)
    plt.ylabel('Relative Abundance (%)', fontsize=16)
    n_total_symbol = r'$N_{total}$'
    n_clonos_symbol = r'$N_{clonotypes}$'
    plt.title(f"{n_total_symbol} = {tcr_df['num_records'].sum()}\n{n_clonos_symbol} = {tcr_df['clonotype_id'].nunique()}")
    plt.tight_layout()  # Adjust layout for better presentation
    if output_filepath:
        plt.savefig(output_filepath, 
                    bbox_inches='tight', 
                    dpi=dpi)
    return plt.show()


def sequence_length_distributions(tcr_df: pd.DataFrame,
                                  seq_len_colnames: list=['alpha_cdr1_length', 
                                                          'alpha_cdr2_length', 
                                                          'alpha_cdr3_length', 
                                                          'beta_cdr1_length', 
                                                          'beta_cdr2_length', 
                                                          'beta_cdr3_length'],
                                  output_filepath: str=None,
                                  figsize: tuple=(15, 10),
                                  dpi: int=450):
    """This function generates a panel of bar plots showing length distributions of various 
    segments of the TCR sequences (default: CDR1,CDR2,CDR3 of both alpha and beta chains)"""
    # Setting up the figure and axes for the subplots
    fig, axs = plt.subplots(2, 3, figsize=figsize)  # 2 rows, 3 columns
    for i, ax in enumerate(axs.flat):
        seq_len_name = seq_len_colnames[i]
        data = (tcr_df
                .groupby(seq_len_name)
                .agg(num_records=('num_records', 'sum'),
                     pct_records=('pct_records', 'sum'))
                .reset_index())
        # Filter out sequence lengths that have < 1% representation in the TCR data
        data = data.loc[data['pct_records']>=0.01].reset_index().copy()
        # Set style and context
        sns.barplot(data=data, x=seq_len_name, 
                    y='num_records', color='lightblue', 
                    ax=ax)
        ax.set_xlabel('')
        ax.set_ylabel('')
        # Add percentages inside each bar
        for index, row in data.iterrows():
            ax.text(index, row.num_records/2, f'{row.pct_records*100:.1f}%', 
                    fontsize=10, ha='center', color='black')
        ax.set_title((seq_len_name.replace('alpha_', r'$\alpha$-')
                                  .replace('beta_', r'$\beta$-')
                                  .replace('cdr', 'CDR')
                                  .replace('_length', '')), fontsize=16)
        # Hide the plot borders
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        # Show only the lines for the x-axis and y-axis
        ax.spines['left'].set_linewidth(2.)
        ax.spines['bottom'].set_linewidth(2.)
        ax.spines['left'].set_color('black')
        ax.spines['bottom'].set_color('black')
    # Setting a common X and Y label and an overall title
    fig.supxlabel('Amino Acid Length', fontsize=18)
    fig.supylabel('Cell (Absolute) Count', fontsize=18)
    plt.tight_layout()  # Adjust layout for better presentation
    if output_filepath:
        plt.savefig(output_filepath, 
                    dpi=dpi, 
                    bbox_inches='tight')
    return plt.show()


def chain_pairing_configurations(tcr_df: pd.DataFrame,
                                 clonotype_definition: list=['cdr1', 'cdr2', 'cdr3'],
                                 figsize: tuple=(8,6),
                                 output_filepath: str=None,
                                 dpi: int=350):
    """This function generates a bar plot showing the different alpha-beta pairing configurations 
    that are present in the TCR data prior to processing (i.e. QC)"""
    # aggregate data by cell ID and chain type (alpha or beta), count number of chains observed 
    tcr_df['chain_identifier'] = tcr_df.apply(lambda row: '_'.join(row[col] for col in clonotype_definition), 
                                              axis=1)
    # tcr_df['chain_identifier'] = tcr_df['cdr1'] + '_' + tcr_df['cdr2'] + '_' + tcr_df['cdr3']
    chain_result1 = (tcr_df
                     .groupby(['barcode', 'chain'])
                     .agg(num_records=('chain_identifier', 'nunique'))
                     .reset_index())
    # pivot table (long-to-wide) cell x chain
    chain_result2 = (chain_result1
                     .pivot_table(index='barcode',
                                  columns='chain', values='num_records')
                     .fillna(0)
                     .astype(int)
                     .reset_index())
    # assign chain-pairing configuration for each unique cell
    chain_result2.loc[(chain_result2['TRA']==1)
                     &(chain_result2['TRB']==1), 'pair_case'] = 'α1-β1'
    chain_result2.loc[(chain_result2['TRA']==2)
                     &(chain_result2['TRB']==1), 'pair_case'] = 'α2-β1'
    chain_result2.loc[(chain_result2['TRA']==1)
                     &(chain_result2['TRB']==2), 'pair_case'] = 'α1-β2'
    chain_result2.loc[(chain_result2['TRA']==2)
                     &(chain_result2['TRB']==2), 'pair_case'] = 'α2-β2'
    chain_result2.loc[(chain_result2['TRA']==0)
                     &(chain_result2['TRB']==1), 'pair_case'] = 'α0-β1'
    chain_result2.loc[(chain_result2['TRA']==1)
                     &(chain_result2['TRB']==0), 'pair_case'] = 'α1-β0'
    # aggregate data by chain pairing configuration, count number of cells observed in each configuration
    chain_result3 = (chain_result2
                     .groupby('pair_case')
                     .agg(cell_count=('barcode', 'nunique'))
                     .reset_index())
    # generate bar plot showing chain pairing configurations
    indices = []
    labels = []
    fig, ax = plt.subplots(figsize=figsize)
    total_count = tcr_df['barcode'].nunique()
    for i, (lab, grp) in enumerate(chain_result3.groupby('pair_case')):
        indices.append(i)
        labels.append(lab)
        # Plot the bar heights
        ax.bar(i, grp['cell_count'].mean(), color='none', edgecolor='black', 
               linewidth=4, label=labels[i])
        ax.annotate(f"{round(grp['cell_count'].sum()*100/total_count,2)}%", (i-0.2, grp['cell_count'].mean()+100))
    # Set the x-axis tick labels
    ax.set_xlabel("α-β pairing configuration", fontsize=16)
    ax.set_xticks(indices)
    ax.set_xticklabels(labels)
    # Set the y-axis label
    ax.set_ylabel(f'Cell (Absolute) Count', fontsize=16)
    # Hide the plot borders
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    # Show only the lines for the x-axis and y-axis
    ax.spines['left'].set_linewidth(2.)
    ax.spines['bottom'].set_linewidth(2.)
    ax.spines['left'].set_color('black')
    ax.spines['bottom'].set_color('black')
    # Remove ticks from the top and right sides of the plot
    ax.tick_params(right=False, top=False)
    # Consolidate the legends
    ax.legend().remove()
    # Adjust layout for better presentation
    plt.tight_layout()
    # Save the plot
    if output_filepath:
        plt.savefig(output_filepath, 
                    dpi=dpi, 
                    bbox_inches='tight')
    return plt.show()


def clonotype_abundances(tcr_df: pd.DataFrame,
                         output_filepath: str=None,
                         figsize: tuple=(10, 6),
                         dpi: int=350):
    """This function generates a histogram showing the distribution of clonotype abundances in the 
    processed TCR data."""
    # Create the histogram
    plt.figure(figsize=figsize)
    total_num_cells = tcr_df['num_records'].sum()
    total_num_clonos = tcr_df['clonotype_id'].nunique()
    data = tcr_df.copy()
    num_bins = int(np.sqrt(len(data['pct_records'])))
    n, bins, patches = plt.hist(data['num_records'], bins=num_bins, color='skyblue', edgecolor='black')
    IQR = np.percentile(data['num_records'], 75) - np.percentile(data['num_records'], 25)
    bin_width = 2 * IQR * (len(data['num_records']) ** (-1/3))
    # Annotate number of clonotypes above each bin
    for i in range(len(n)):
        # Calculate the center of each bin
        bin_center = (bins[i] + bins[i+1]) / 2
        plt.text(bin_center, n[i], str(int(n[i])), ha='center', va='bottom')
    # Remove top and right borders
    ax = plt.gca()  # Get current axes
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    # Show only the lines for the x-axis and y-axis
    ax.spines['left'].set_linewidth(2.)
    ax.spines['bottom'].set_linewidth(2.)
    ax.spines['left'].set_color('black')
    ax.spines['bottom'].set_color('black')
    # Adding labels to the plot
    plt.xlabel('Absolute Abundance', fontsize=16)
    plt.ylabel('Clonotype Count', fontsize=16)
    n_total_symbol = r'$N_{total}$'
    n_clonos_symbol = r'$N_{clonotypes}$'
    plt.title(f'{n_total_symbol} = {total_num_cells}\n{n_clonos_symbol} = {total_num_clonos}')
    plt.grid(False)
    # Save the plot
    if output_filepath:
        plt.savefig(output_filepath, 
                    dpi=dpi, 
                    bbox_inches='tight')
    return plt.show()