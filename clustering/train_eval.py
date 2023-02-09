from argparse import ArgumentParser
from pathlib import Path
import pandas as pd
from pickle import load, dump
from typing import List, Dict, Tuple
from collections import OrderedDict
import numpy as np
from sklearn.preprocessing import QuantileTransformer

# from model import RegressionHyperModel

input_cols = ['R [Rsun]', 'B [G]', 'alpha [deg]']
output_cols = ['n [cm^-3]', 'v [km/s]', 'T [MK]']
    
def load_original_data(data_path: Path) -> Dict[str, pd.DataFrame]:
    """Load the original data from the file."""
    
    filenames = [f for f in data_path.iterdir()]
    all_data_df = pd.DataFrame(columns=input_cols+output_cols)
    for f in filenames:
        all_data_df = pd.concat([all_data_df, 
                   pd.read_csv(f, skiprows=2, usecols=input_cols+output_cols)], axis=0)
        
    scaler = QuantileTransformer()
    scaler.fit(all_data_df)
    all_data_df = scaler.transform(all_data_df)
    all_data_df = pd.DataFrame(all_data_df, columns=input_cols+output_cols)
        
    # separate into file series
    all_data_df = [all_data_df.iloc[i*640:i*640+640, :] for i in range(len(all_data_df) // 640)]
    
    return OrderedDict({f.stem : df for f, df in zip(filenames, all_data_df)})

def join_files_in_cluster(cluster_files: List[Path], input_cols: List[str], output_cols: List[str]) -> pd.DataFrame:
    """Join all files in a cluster into a single dataframe."""
    cluster_df = pd.DataFrame(columns=[input_cols+output_cols])
    
    for f in cluster_files:
        df = pd.read_csv(f, skiprows=2, usecols=input_cols+output_cols)
        cluster_df = pd.concat([cluster_df, df], axis=0)

    scaler = QuantileTransformer()
    inputs_df = cluster_df[[input_cols]]
    inputs_df = scaler.fit_transform(inputs_df)
    
    scaler = QuantileTransformer()
    outputs_df = cluster_df[[output_cols]]
    outputs_df = scaler.fit_transform(outputs_df)
    
    return inputs_df, outputs_df

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--data', '-d', type=str, required=True)
    parser.add_argument('--clusters-files', '-c', type=str, default='./conf')
    
    opts = vars(parser.parse_args())
    
    conf_files = [f for f in Path(opts['clusters_files']).iterdir() if f.is_file()]
    
    print('Preparing data...')
    data_dict = load_original_data(Path(opts['data']))
    
    for f in conf_files:
        print(f'Running {f}...')
        
        with open(f, 'rb') as cf:
            cluster_runs = load(cf)
    
        # test each run
        for run in cluster_runs:
            # train model on each cluster
            for cluster_id, cluster in run.items():
                
                cluster_inputs, cluster_outputs = join_files_in_cluster(cluster, input_cols, output_cols)

                model = RegressionHyperModel()

                
            
            
                
            
            
        
        
        
    
    