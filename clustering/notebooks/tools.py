import pickle as pkl
import math
import matplotlib.pyplot as plt
import numpy as np
from scipy.cluster.hierarchy import dendrogram
import seaborn as sns
import pandas as pd

def plot_km_results(cluster_count, labels, series, save_path=None, scale="log"):
    plot_count = math.ceil(math.sqrt(cluster_count))
    fig, axs = plt.subplots(plot_count,plot_count,figsize=(10,10), sharey="all")
    fig.suptitle('Clusters')
    row_i=0
    column_j=0
    # For each label there is,
    # plots every series with that label
    for label in set(labels):
        cluster = []
        for i in range(len(labels)):
            if(labels[i]==label):
                axs[row_i, column_j].plot(series[i],c="gray",alpha=0.4)
                cluster.append(series[i])
        if len(cluster) > 0:
            axs[row_i, column_j].plot(np.average(np.vstack(cluster),axis=0),c="red")
        axs[row_i, column_j].set_title("Cluster "+str(row_i*plot_count+column_j))
        column_j+=1
        if column_j%plot_count == 0:
            row_i+=1
            column_j=0
    plt.tight_layout()
    
    for ax in axs.flat:
        ax.set_yscale(scale)
    
    if save_path != None:
        plt.savefig(save_path, dpi=200)
        
    plt.show()
    
 
    
def plot_unscaled_clusters(labels, nclusters, flows_dict, columns, yscale={}, save_path=None):
    """Plot cluster of the original data (not scaled)"""  
    
    if (nclusters < 2):
        print("Not enough clusters to plot")
        return
    
    _fig, axs = plt.subplots(nrows=nclusters, ncols=len(columns), figsize=(4*len(columns), 4*nclusters), sharey="col")
    for idx, flow in enumerate(flows_dict.values()):
        for col_pos, col in enumerate(columns):
            axs[(labels[idx], col_pos)].plot(flow[col], linewidth=0.5)
            axs[(labels[idx], col_pos)].set(ylabel=col, yscale=yscale.get(col, 'linear'))
        axs[(labels[idx], 1)].set_title("Cluster " + str(labels[idx]))
                 
    plt.tight_layout()
    
    # TODO share y along columns
    
    if save_path != None:
        plt.savefig(save_path, dpi=200)
    plt.show()


  
def plot_dendrogram(model, **kwargs):
    # Create linkage matrix and then plot the dendrogram
    # create the counts of samples under each node
    counts = np.zeros(model.children_.shape[0])
    n_samples = len(model.labels_)
    for i, merge in enumerate(model.children_):
        current_count = 0
        for child_idx in merge:
            if child_idx < n_samples:
                current_count += 1  # leaf node
            else:
                current_count += counts[child_idx - n_samples]
        counts[i] = current_count

    linkage_matrix = np.column_stack([model.children_, model.distances_,
                                      counts]).astype(float)

    # Plot the corresponding dendrogram
    dendrogram(linkage_matrix, **kwargs)
    
def plot_cluster_file_group(filenames, labels, nclusters):
    # get filegroups and create a dict with the number of files per cluster
    locs = set([l.stem.split('_')[2] for l in filenames])
    cluster_file_distr = {loc:[0 for _ in range(nclusters)] for loc in locs}

    # count the number of files per cluster
    for f, c in zip(filenames, labels):
        loc = f.stem.split('_')[2]
        cluster_file_distr[loc][c] += 1
    cluster_file_distr = dict(sorted(cluster_file_distr.items()))
    print(cluster_file_distr)
    
    # arrange data for plotting
    cluster_ids = sorted(list(set(labels)))
    # move -1 to the end in the case of dbscan
    if -1 in cluster_ids:
        cluster_ids = cluster_ids[1:] + [-1]

    print(cluster_ids)
    tmp_dict = {"group" : [], "cluster" : [], "count" : []}
    for f, v in cluster_file_distr.items():
        for idx, count in zip(cluster_ids, v):
            tmp_dict["group"].append(f)
            tmp_dict["cluster"].append(idx)
            tmp_dict["count"].append(count)
            
    fig, ax = plt.subplots(figsize=(10, 5))       
    sns.barplot(x="group", y="count", hue="cluster", data=pd.DataFrame(tmp_dict), ax=ax)
    ax.set_xlabel("File Group")
    ax.set_ylabel("Number of files")
    ax.set_title("Number of files per cluster per file group")
    fig.legend(loc="center right", title="Cluster")
    ax.get_legend().remove()
    return fig