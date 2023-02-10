from argparse import ArgumentParser
from pathlib import Path
import pandas as pd
from pickle import load, dump
from typing import List, Dict, Tuple
from collections import OrderedDict
import numpy as np
from sklearn.preprocessing import QuantileTransformer
from sklearn.model_selection import train_test_split

import logging

from keras_tuner import RandomSearch

from model import RegressionHyperModel

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

def join_files_in_cluster(cluster_files: List[Path], original_data : Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Join all files in a cluster into a single dataframe."""
    cluster_inputs, cluster_outputs = pd.DataFrame(), pd.DataFrame()
    
    print("cluster size", len(cluster_files))
    for idx, f in enumerate(cluster_files):
        cluster_inputs = pd.concat([cluster_inputs, original_data[f].iloc[:, 0:3]], axis=1, ignore_index=True)
        cluster_outputs = pd.concat([cluster_outputs, original_data[f].iloc[:, 3:]], axis=1, ignore_index=True)        
    
    print(cluster_inputs.head())
    print(cluster_inputs.shape)
    # print(cluster_df)
    # print(cluster_df.shape)
    # print(cluster_df.columns)

    return cluster_inputs, cluster_outputs

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--data', '-d', type=Path, required=True)
    parser.add_argument('--model-output', '-m', type=Path, required=True)
    parser.add_argument('--clusters-files', '-c', type=str, default='./conf')
    
    opts = vars(parser.parse_args())
    
    conf_files = [f for f in Path(opts['clusters_files']).iterdir() if f.is_file()]
    
    if not opts['model_output'].exists():
        opts['model_output'].mkdir(parents=True, exist_ok=False)
    
    print('Preparing data...')
    data_dict = load_original_data(opts['data'])
    
    for f in conf_files:
        print(f'Running {f}...')
        
        with open(f, 'rb') as cf:
            cluster_runs = load(cf)
            
        out_dir = opts['model_output'] / f.stem
        if not out_dir.exists():
            out_dir.mkdir(parents=True, exist_ok=False)
        
        logging.basicConfig(filename=out_dir / 'train.log', format='%(asctime)s - %(levelname)s - %(message)s')
    
        # test each run
        for run_id, run in enumerate(cluster_runs):
            # train model on each cluster
            for cluster_id, cluster in run.items():
                
                try:
                    cluster_inputs, cluster_outputs = join_files_in_cluster(cluster, data_dict)
                    trainX, testX, trainY, testY = train_test_split(cluster_inputs, cluster_outputs, test_size=0.15, random_state=1)
                    
                    print(trainX.shape, trainY.shape)

                    hypermodel = RegressionHyperModel((trainX.shape[1],))
                    
                    print((trainX.shape[1],))
                    
                    tuner_rs = RandomSearch(
                        hypermodel,
                        objective='mse',
                        seed=42,
                        max_trials=10,
                        executions_per_trial=1,
                        overwrite=True
                    )

                    tuner_rs.search(trainX, trainY, epochs=500, validation_split=0.2, verbose=1)  #epochs=500

                    best_model_r = tuner_rs.get_best_models(num_models=1)[0]
                    loss_r, mse_r = best_model_r.evaluate(testX, testY)
                    print(f"Random Search - {f.stem} : {cluster_id}")
                    print(best_model_r.summary())
                    print(loss_r,mse_r)
                    
                    logging.info(f'{f.stem}_run{run_id}_c{cluster_id} - loss: {loss_r} - mse: {mse_r}')
                    logging.info(f'{f.stem}_run{run_id}_c{cluster_id} - {best_model_r.summary()}')
                    
                    best_model_r.save(out_dir / f'{f.stem}_{cluster_id}.h5')
                except Exception as e:
                    logging.error(f'Error in {f.stem}_run{run_id}_c{cluster_id}', exc_info=True)
                    exit(1)

                
            
            
                
            
            
        
        
        
    
    