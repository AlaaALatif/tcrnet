import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
from .networks.graph import compute_node_positions
from .networks.metrics import compute_network_metrics


def generate_network_plot(graph: nx.classes.graph.Graph,
                          network_metrics: dict,
                          partition: dict,
                          colors: dict,
                          output_filepath: str=None,
                          k: float=.15,
                          seed: int=42,
                          plot_title: str='',
                          figsize: tuple=(12, 10),
                          dpi: int=500,
                          nx_kwargs: dict=None):
    # set the desired figure resolution
    plt.rcParams['figure.dpi'] = dpi
    if not nx_kwargs:
        nx_kwargs = {"edgecolors": "tab:gray"}
    # Set the desired figure size (adjust width and height as needed)
    fig = plt.figure(figsize=figsize)
    node_positions = compute_node_positions(graph, k=k, seed=seed)
    # Create lists of nodes in the graph
    nodes = [node for node in graph.nodes]
    # Draw the network graphs with circles for the 1st state and triangles for the 2nd state
    nx.draw(graph,
            nodelist=nodes,  # Only nodes in the 1st state
            pos=node_positions,
            node_color=[colors.get(partition.get(i), 'grey') for i in nodes],
            node_shape='o',  # Circle shape for 1st state
            node_size=50,
            with_labels=False,
            **nx_kwargs)
    # Annotate each cluster with its cluster number
    for cluster, color in list(colors.items()):
        cluster_nodes = [node for node, part in partition.items() if part == cluster]
        x, y = zip(*[node_positions[node] for node in cluster_nodes])
        x_center, y_center = sum(x) / len(cluster_nodes), sum(y) / len(cluster_nodes)
        plt.text(x_center, y_center, f'{cluster}', fontsize=11, 
                 color='black', ha='center', va='center', fontweight='bold')
    # legend customization
    ax = plt.gca()
    # generate title for plot
    latex_symbol1 = r'$D_{total}$'
    latex_symbol2 = r'$\overline{S}(C_{x})_{x \in [0:12]}$'
    latex_symbol3 = r'$\overline{S}(C_{x}, C_{y})_{x,y \in [0:12], x \neq y}$'
    latex_symbol4 = r'$\beta$'
    plt.title(f"""{plot_title}\n
              {latex_symbol1} = {network_metrics['total_density']:.6f}\n
              {latex_symbol2} = {np.mean(network_metrics['intracluster_density']):.4G}\n
              {latex_symbol3} = {np.mean(network_metrics['intercluster_density']):.4G}""")
    if output_filepath:
        plt.savefig(output_filepath, 
                    bbox_inches='tight',
                    dpi=dpi)
    return plt.show()


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
                 f"{row['clonotype_id']} ({row['pct_records']*100:.3f}%)", 
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