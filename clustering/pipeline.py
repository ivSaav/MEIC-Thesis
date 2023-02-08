from pathlib import Path
from argparse import ArgumentParser
from collections import OrderedDict
import pandas as pd
import numpy as np

# preprocessing
from sklearn.preprocessing import StandardScaler, MinMaxScaler, MaxAbsScaler, RobustScaler, QuantileTransformer
from sklearn.decomposition import PCA

# clustering
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from tslearn.clustering import TimeSeriesKMeans
from minisom import MiniSom

def time_series_kmeans(compiled_df : np.ndarray, flows_dict : dict, max_clusters : int = 12):
    print("Time Series K-Means")
    # scale data
    quantile_scaler = QuantileTransformer(n_quantiles=1000, output_distribution='uniform')
    quantile_scaler.fit(compiled_df)
    scaled_flows = quantile_scaler.transform(compiled_df)
    
    # ============ MAGNETIC FIELD ============
    # separate into file series
    scaled_magnetic = [scaled_flows['B [G]'][i*640 : i*640 + 640] for i in range(len(scaled_flows['B [G]']) // 640)]
    scaled_magnetic = np.array(scaled_magnetic)

    mag_clusters = []
    for i in range(2, max_clusters + 1):
        km = TimeSeriesKMeans(n_clusters=i, metric="euclidean", max_iter=500, random_state=0)
        mag_labels = km.fit_predict(scaled_magnetic)
        
        # organize filenames into clusters
        mag_cluster = {k : list() for k in range(i)} 
        for cluster_label, flow in zip(mag_labels, flows_dict.keys()):
            mag_cluster[cluster_label].append(flow)   
        mag_clusters.append(mag_cluster)
    
    # ============ INCLINATION ============
    # separate into file series
    # separate into file series
    scaled_alpha = [scaled_flows['alpha [deg]'][i*640 : i*640 + 640] for i in range(len(scaled_flows['alpha [deg]']) // 640)]
    scaled_alpha = np.array(scaled_alpha)
    
    alpha_clusters = []
    for i in range(2, max_clusters + 1):
        km = TimeSeriesKMeans(n_clusters=i, metric="euclidean", max_iter=500, random_state=0)
        alpha_labels = km.fit_predict(scaled_alpha)
        
        # organize filenames into clusters
        alpha_cluster = {k : list() for k in range(i)} 
        for cluster_label, flow in zip(alpha_labels, flows_dict.keys()):
            mag_cluster[cluster_label].append(flow)   
        alpha_clusters.append(alpha_cluster)
    
    return mag_clusters, alpha_clusters    
    # TODO : save clusters to file
    
#   def time_series_som()      
    

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--data', '-d', type=Path, required=True)
    # parser.add_argument('--output', '-o', type=Path, required=True)
    parser.add_argument('--methods', '-m', type=str, nargs='+', 
                        default=['minisom', 'ts_kmeans', 
                                 'pca_kmeans', 'pca_agg', 'pca_dbscan', 
                                 'pca_joint_kmeans', 'pca_joint_agg', 'pca_joint_dbscan'])
    parser.add_argument('--scalers', '-s', type=str, nargs='+',
                        default=['standard', 'minmax', 'maxabs', 'robust', 'quantile'])
    parser.add_argument('--max-clusters', type=int, default=12)
    parser.add_argument('--min-cluster-size', type=int, default=100)
    
    args = parser.parse_args()
    opts = vars(args)
    
    # load data
    filenames = [f for f in opts['data'].iterdir()]
    flows_dict = OrderedDict()
    compiled_df = pd.DataFrame(columns=['R [Rsun]', 'B [G]', 'alpha [deg]'])
    for f in filenames:
        flows_dict[f.stem] = pd.read_csv(f, skiprows=2, usecols=['R [Rsun]', 'B [G]', 'alpha [deg]'])
        compiled_df = pd.concat([compiled_df, flows_dict[f.stem]], axis=0)


    if 'ts_means' in opts['methods']:
        time_series_kmeans(compiled_df, flows_dict, opts['max_clusters'])
        
        
    