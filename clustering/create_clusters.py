from pathlib import Path
from argparse import ArgumentParser
from collections import OrderedDict
import pandas as pd
import pickle as pkl

# preprocessing
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, QuantileTransformer, MaxAbsScaler
from sklearn.decomposition import PCA

from methods import *
    
def save_normalized_clusters(run_clusters, min_cluster_size, method_name, save_path=None, outliers=False):
    """Remove or  with less than min_cluster_size members."""
    final = []
    for run_id, run in enumerate(run_clusters):
        # remove outliers from dbscan
        if outliers:
            run.pop(-1)
            
        lengths = [len(run[k]) for k in run]
        print(lengths)
        
        remove_set = set()
        # remove strange clusters with 1 value
        for k, length in zip(run.keys(), lengths):
            if length == 1:
                remove_set.add(k)
        for k in remove_set:
            run.pop(k)
        lengths = [len(run[k]) for k in run]
        
        # exclude any that do not have necessary min_clusters or only one cluster
        if len(lengths) > 1 and min(lengths) >= min_cluster_size:
            run_dict = {
                'run_id': run_id,
                'method': method_name,
                'clusters': run,
            }
            final.append(run_dict)   
    
    print(f'Number of runs: {len(final)}')      
    if (len(final) > 0):
        pkl.dump(final, open(save_path, 'wb'))

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--data', '-d', type=Path, required=True)
    parser.add_argument('--output', '-o', type=Path, default='./config')
    parser.add_argument('--methods', '-m', type=str, nargs='+', 
                        default=['minisom', 'ts_kmeans', 
                                 'pca_kmeans', 'pca_agg', 'pca_dbscan', 
                                 'pca_joint_kmeans', 'pca_joint_agg', 'pca_joint_dbscan'])
    parser.add_argument('--max-clusters', type=int, default=10)
    parser.add_argument('--min-cluster-size', type=int, default=2000)
    
    args = parser.parse_args()
    opts = vars(args)
    
    if not opts['output'].exists():
        opts['output'].mkdir(parents=True, exist_ok=True)
    
    flow_columns = ['R [Rsun]', 'B [G]', 'alpha [deg]']    
    scalers = {
        'standard': StandardScaler(),
        'minmax': MinMaxScaler((-1,1)),
        'maxabs': MaxAbsScaler(),
        'robust': RobustScaler(),
        'quantile': QuantileTransformer()
    }
    
    ts_methods = {
        'ts_kmeans': time_series_kmeans,
        'minisom': time_series_som,
    }
    
    pca_methods = {
        'pca_kmeans' : kmeans,
        'pca_agg' : agg,
        'pca_dbscan' : dbscan
    }
    
    # Load data
    print('Loading data...')
    filenames = [f for f in opts['data'].iterdir()]
    flows_dict = OrderedDict()
    for f in filenames:
        flows_dict[f.stem] = pd.read_csv(f, skiprows=2, usecols=['R [Rsun]', 'B [G]', 'alpha [deg]'])
    
    compiled_df = list(flows_dict.values())   
    compiled_df = pd.concat(compiled_df, axis=0)
    compiled_df.reset_index(drop=True, inplace=True)
    compiled_df.columns = ['R [Rsun]', 'B [G]', 'alpha [deg]']
    
    filenames = list(flows_dict.keys())

    # Scale data
    quantile_scaler = QuantileTransformer(n_quantiles=1000, output_distribution='uniform')
    quantile_scaler.fit(compiled_df)
    quant_scaled_flows = quantile_scaler.transform(compiled_df)
    quant_scaled_flows = pd.DataFrame(quant_scaled_flows, columns=compiled_df.columns)
    
    quant_scaled_mag = [quant_scaled_flows['B [G]'][i*640 : i*640 + 640] for i in range(len(quant_scaled_flows) // 640)]
    quant_scaled_mag = np.array(quant_scaled_mag) 
    
    quant_scaled_alpha = [quant_scaled_flows['alpha [deg]'][i*640 : i*640 + 640] for i in range(len(quant_scaled_flows) // 640)]
    quant_scaled_alpha = np.array(quant_scaled_alpha)
    
    # Time Series methods
    for name, method in ts_methods.items():
        print(f'Running {name}...')
        mag_runs = method(quant_scaled_mag, filenames, opts['max_clusters'])
        save_normalized_clusters(mag_runs, opts['min_cluster_size'], f'{name}_mag', opts['output'] / f'{name}_mag.pkl')
        
        alpha_runs = method(quant_scaled_alpha, filenames, opts['max_clusters'])
        save_normalized_clusters(alpha_runs, opts['min_cluster_size'], f'{name}_alpha', opts['output'] / f'{name}_alpha.pkl')
            
    # ==== PCA CLUSTERING ====
    for name, method in pca_methods.items():
        print(f'Running {name}...')
        mag_runs = method(quant_scaled_mag, filenames, opts['max_clusters'])
        save_normalized_clusters(mag_runs, opts['min_cluster_size'], f'{name}_mag', 
                                 opts['output'] / f'{name}_mag.pkl', outliers=name=='pca_dbscan')
        
        alpha_runs = method(quant_scaled_alpha, filenames, opts['max_clusters'])
        save_normalized_clusters(alpha_runs, opts['min_cluster_size'], f'{name}_alpha',
                                 opts['output'] / f'{name}_alpha.pkl', outliers=name=='pca_dbscan')
        
        
    # ==== JOINT PCA CLUSTERING ====
    for scaler_name, scaler in scalers.items():
        # concat flows
        scaled_flows = list(flows_dict.values())
        scaled_flows = pd.concat(scaled_flows, axis=0)

        scaler.fit(scaled_flows)
        scaled_flows = scaler.transform(scaled_flows)
        scaled_flows = pd.DataFrame(scaled_flows, columns=flow_columns)
        
        # separate into file series
        scaled_all = [scaled_flows.iloc[i*640:i*640+640, :] for i in range(len(scaled_flows) // 640)]
        scaled_all = [flow.values for flow in scaled_all]
        scaled_all = np.array([flow.ravel() for flow in np.array(scaled_all)])
        
        # reduce to 2 components
        pca = PCA(n_components=2)
        transformed = pca.fit_transform(scaled_all)
        
        for name, method in pca_methods.items():
            print(f'Running {name} with {scaler_name}...')
            runs = method(transformed, filenames, opts['max_clusters'])     
            save_normalized_clusters(runs, opts['min_cluster_size'], f'{scaler_name}_{name}',
                                     opts['output'] / f'{scaler_name}_{name}.pkl', outliers=name=='pca_dbscan')
        
        
    