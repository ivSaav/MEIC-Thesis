from collections import OrderedDict
import numpy as np
import math
import pandas as pd

from typing import List, Dict, Tuple

# clustering
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.decomposition import PCA
from tslearn.clustering import TimeSeriesKMeans
from minisom import MiniSom

def map_filename_to_cluster(labels, filenames):
    """Organizes the filenames into clusters"""
    clusters = {k : list() for k in set(labels)} 
    for cluster_label, filename in zip(labels, filenames):
        clusters[cluster_label].append(filename)
    return clusters

def time_series_kmeans(scaled_files : np.ndarray, filenames : dict, max_clusters : int = 12) -> List[Dict[int, List[str]]]:
    # generate all possiblle cluster
    cluster_runs = []
    for i in range(2, max_clusters + 1):
        km = TimeSeriesKMeans(n_clusters=i, metric="euclidean", max_iter=500, random_state=0)
        labels = km.fit_predict(scaled_files)
        cluster_runs.append(map_filename_to_cluster(labels, filenames))
    
    return cluster_runs   
    
    
def time_series_som(scaled_files : np.ndarray, filenames : dict, max_clusters : int = 12) -> List[Dict[int, List[str]]]:
    # generate all possiblle cluster
    cluster_runs = []
    train_x = list(scaled_files)
    for i in range(2, math.ceil(math.sqrt(max_clusters + 1))):
        som = MiniSom(i, i, 640, sigma=0.3, learning_rate = 0.1)
        som.random_weights_init(train_x)
        som.train(train_x, 50000, verbose=True) 
        
        clusters = OrderedDict()
        for scaled, filename in zip(scaled_files, filenames):
            winner_node = som.winner(scaled)
            clusters.setdefault(winner_node[0]*i + winner_node[1] + 1, list()).append(filename)
            
        cluster_runs.append(clusters)
    return cluster_runs

def kmeans(scaled_files : np.ndarray, filenames : dict, max_clusters : int = 12) -> List[Dict[int, List[str]]]:  
    # generate all possiblle cluster
    cluster_runs = []
    for i in range(2, max_clusters + 1):
        km = KMeans(n_clusters=i, random_state=0, n_init='auto', max_iter=500)
        labels = km.fit_predict(scaled_files)
        cluster_runs.append(map_filename_to_cluster(labels, filenames))
    
    return cluster_runs


def agg(scaled_files : np.ndarray, filenames : dict, max_clusters : int = 12) -> List[Dict[int, List[str]]]:
    # generate all possiblle cluster
    cluster_runs = []
    for i in range(2, max_clusters + 1):
        km = AgglomerativeClustering(n_clusters=i)
        labels = km.fit_predict(scaled_files)
        cluster_runs.append(map_filename_to_cluster(labels, filenames))
    
    return cluster_runs


def dbscan(scaled_files : np.ndarray, filenames : dict, max_clusters : int = 12) -> List[Dict[int, List[str]]]:
    
    # generate all possiblle cluster
    cluster_runs = []
    # TODO work with different ranges for minsamples
    for i in np.arange(0.1, 0.5, 0.05):
        km = DBSCAN(eps=i, min_samples=10, metric='euclidean', n_jobs=-1)
        labels = km.fit_predict(scaled_files)
        ncluster = len(set(labels)) - 1
        
        if ncluster > max_clusters:
            continue
        
        cluster_runs.append(map_filename_to_cluster(labels, filenames))
    return cluster_runs

    
    
    
    
    