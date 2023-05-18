import pickle as pkl
import math
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples, silhouette_score
import numpy as np
from scipy.cluster.hierarchy import dendrogram
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score, fowlkes_mallows_score, rand_score
import seaborn as sns
import pandas as pd

from pickle import dump
from typing import List
from pathlib import Path
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


def clustering_metrics(model, data, params) -> pd.DataFrame:
    scores = {"n_clusters" : [], "silhouette" : [], "davies_bouldin" : [], "calinski_harabasz" : [],}
    for i in range(2, 10):
        kmeans = model(n_clusters=i, **params)
        labels = kmeans.fit_predict(data)
        scores["n_clusters"].append(i)
        scores["silhouette"].append(silhouette_score(data, labels))
        scores["davies_bouldin"].append(davies_bouldin_score(data, labels))
        # scores["rand"].append(rand_score(data, labels))
        scores["calinski_harabasz"].append(calinski_harabasz_score(data, labels))
        # scores["fowlkes_mallows"].append(fowlkes_mallows_score(data, labels))
    return pd.DataFrame(scores, index=None)  


def silhouette_plot(X, nclusters, model, params):
    for n_clusters in range(2, nclusters+1):
        # Create a subplot with 1 row and 2 columns
        fig, ax1 = plt.subplots(figsize=(5,5))

        # The 1st subplot is the silhouette plot
        # The silhouette coefficient can range from -1, 1 but in this example all
        # lie within [-0.1, 1]
        ax1.set_xlim([-0.1, 1])
        # The (n_clusters+1)*10 is for inserting blank space between silhouette
        # plots of individual clusters, to demarcate them clearly.
        ax1.set_ylim([0, len(X) + (n_clusters + 1) * 10])

        # Initialize the clusterer with n_clusters value and a random generator
        # seed of 10 for reproducibility.
        clusterer = model(n_clusters=n_clusters, **params)
        cluster_labels = clusterer.fit_predict(X)

        # The silhouette_score gives the average value for all the samples.
        # This gives a perspective into the density and separation of the formed
        # clusters
        silhouette_avg = silhouette_score(X, cluster_labels)
        print(
            "For n_clusters =",
            n_clusters,
            "The average silhouette_score is :",
            silhouette_avg,
        )

        # Compute the silhouette scores for each sample
        sample_silhouette_values = silhouette_samples(X, cluster_labels)

        y_lower = 10
        for i in range(n_clusters):
            # Aggregate the silhouette scores for samples belonging to
            # cluster i, and sort them
            ith_cluster_silhouette_values = sample_silhouette_values[cluster_labels == i]

            ith_cluster_silhouette_values.sort()

            size_cluster_i = ith_cluster_silhouette_values.shape[0]
            y_upper = y_lower + size_cluster_i

            color = cm.nipy_spectral(float(i) / n_clusters)
            ax1.fill_betweenx(
                np.arange(y_lower, y_upper),
                0,
                ith_cluster_silhouette_values,
                facecolor=color,
                edgecolor=color,
                alpha=0.7,
            )

            # Label the silhouette plots with their cluster numbers at the middle
            ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))

            # Compute the new y_lower for next plot
            y_lower = y_upper + 10  # 10 for the 0 samples

        ax1.set_title(f"Silhouette plot ({i} clusters)")
        ax1.set_xlabel("The silhouette coefficient values")
        ax1.set_ylabel("Cluster label")

        # The vertical line for average silhouette score of all the values
        ax1.axvline(x=silhouette_avg, color="red", linestyle="--")

        ax1.set_yticks([])  # Clear the yaxis labels / ticks
        ax1.set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])

def save_cluster_info(labels : List[int], filenames : List[str], run_id : int, method_name : str, save_path : Path):

    clusters = {i : [] for i in set(labels)}
    for f, c in zip(filenames, labels):
        clusters.get(c, [])
        clusters[c].append(f)
    run_dict = {
                'run_id': run_id,
                'method': method_name,
                'clusters': clusters,
                'small_clusters': [],
    }
    
    run = {run_id : run_dict}

    dump(run, open(save_path / f"{method_name}.pkl", "wb"))