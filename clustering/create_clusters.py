from pathlib import Path
from argparse import ArgumentParser
from collections import OrderedDict
import pandas as pd

# preprocessing
from sklearn.preprocessing import StandardScaler, MinMaxScaler, MaxAbsScaler, RobustScaler, QuantileTransformer
from sklearn.decomposition import PCA

from methods import *

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
    parser.add_argument('--max-clusters', type=int, default=6)
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
    compiled_df.reset_index(drop=True, inplace=True)


    # ==== TIME SERIES CLUSTERING ====
    # scale data
    quantile_scaler = QuantileTransformer(n_quantiles=1000, output_distribution='uniform')
    quantile_scaler.fit(compiled_df)
    quant_scaled_flows = quantile_scaler.transform(compiled_df)
    quant_scaled_flows = pd.DataFrame(quant_scaled_flows, columns=compiled_df.columns)
    
    if 'ts_kmeans' in opts['methods']:
        mag_runs = time_series_kmeans(quant_scaled_flows, 'B [G]', filenames, opts['max_clusters'])
        
        alpha_runs = time_series_kmeans(quant_scaled_flows, 'alpha [deg]', filenames, opts['max_clusters'])
        
        # TODO save to file
        
    if 'minisom' in opts['methods']:
        mag_runs = time_series_som(quant_scaled_flows, 'B [G]', filenames, opts['max_clusters'])
        alpha_runs = time_series_som(quant_scaled_flows, 'alpha [deg]', filenames, opts['max_clusters'])
        
    
    # ==== PCA CLUSTERING ====
    if 'pca_kmeans' in opts['methods']:
        mag_runs = pca_kmeans(quant_scaled_flows, 'B [G]', filenames, opts['max_clusters'])
        alpha_runs = pca_kmeans(quant_scaled_flows, 'alpha [deg]', filenames, opts['max_clusters'])
        
        # TODO save to file
        
    if 'pca_agg' in opts['methods']:
        mag_runs = pca_agg(quant_scaled_flows, 'B [G]', filenames, opts['max_clusters'])
        alpha_runs = pca_agg(quant_scaled_flows, 'alpha [deg]', filenames, opts['max_clusters'])
        
        # TODO save to file
        
    if 'pca_dbscan' in opts['methods']:
        mag_runs = pca_dbscan(quant_scaled_flows, 'B [G]', filenames, opts['max_clusters'])
        alpha_runs = pca_dbscan(quant_scaled_flows, 'alpha [deg]', filenames, opts['max_clusters'])
        
        # TODO save to file
    
    # ==== JOINT PCA CLUSTERING ====
    # TODO scaler loop for joint analysis
        
        
    