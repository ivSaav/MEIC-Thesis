from argparse import ArgumentParser
from pathlib import Path
import pandas as pd
from pickle import load, dump
import json
from typing import List, Dict, Tuple
from collections import OrderedDict
import numpy as np
from sklearn.preprocessing import QuantileTransformer
from sklearn.model_selection import train_test_split
import gc
from keras import backend as K

from keras.callbacks import EarlyStopping

early_stop = EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True)

import logging

from keras_tuner import RandomSearch

from model import RegressionHyperModel

input_cols = ['R [Rsun]', 'B [G]', 'alpha [deg]']
output_cols = ['n [cm^-3]', 'v [km/s]', 'T [MK]']
    
def load_original_data(data_path: Path) -> Dict[str, pd.DataFrame]:
    """Load the original data from the file."""
    
    inputs = pd.read_csv(data_path / 'inputsdata_compilation.csv')
    inputs.rename({'Unnamed: 0': 'filename'}, axis=1, inplace=True)
    outputs = pd.read_csv(data_path / 'outputsdata_compilation.csv')
    outputs.rename({'Unnamed: 0': 'filename'}, axis=1, inplace=True)    
    
    input_filenames = inputs[['filename']]
    output_filenames = outputs[['filename']]
    
    scaler_inputs, scaler_ouputs = QuantileTransformer(), QuantileTransformer()
    inputs = scaler_inputs.fit_transform(inputs.iloc[:, 1:])
    outputs = scaler_ouputs.fit_transform(outputs.iloc[:, 1:])
    
    inputs = pd.DataFrame(inputs)
    inputs = pd.concat([input_filenames, inputs], axis=1)
    
    outputs = pd.DataFrame(outputs)
    outputs = pd.concat([output_filenames, outputs], axis=1)
    
    print("Scaled inputs:", inputs.head())
    print("Scaled outputs:", outputs.head())
    return inputs, outputs

def join_files_in_cluster(cluster_files: List[Path], input_data : pd.DataFrame, output_data : pd.DataFrame) -> pd.DataFrame:
    """Join all files in a cluster into a single dataframe."""
    cluster_inputs, cluster_outputs = pd.DataFrame(), pd.DataFrame()
    
    inputs = [input_data.loc[input_data['filename'] == f].iloc[:, 1:]
              for f in cluster_files]
    
    cluster_inputs = pd.concat(inputs, axis=0, ignore_index=True)
    
    outputs = [output_data.loc[output_data['filename'] == f].iloc[:, 1:]
               for f in cluster_files]
    cluster_outputs = pd.concat(outputs, axis=0, ignore_index=True)
    # for f in cluster_files:
    #     cluster_inputs = pd.concat(
    #                         [
    #                             cluster_inputs, 
    #                             input_data.loc[input_data['filename'] == f].iloc[:, 1:]
    #                         ],
    #                         axis=0, ignore_index=True)
    #     cluster_outputs = pd.concat(
    #                         [
    #                             cluster_outputs, 
    #                             output_data.loc[output_data['filename'] == f].iloc[:, 1:]
    #                         ], 
    #                         axis=0, ignore_index=True)        
    
    print(cluster_inputs.head())
    print(cluster_inputs.shape)
    # print(cluster_df)
    # print(cluster_df.shape)
    # print(cluster_df.columns)
    print("Cluster shape:", cluster_inputs.shape)

    # TODO val split
    return cluster_inputs, cluster_outputs

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--data', '-d', type=Path, required=True)
    parser.add_argument('--model-output', '-m', type=Path, required=True)
    parser.add_argument('--cluster-files-dir', '-c', type=str)
    parser.add_argument('--cluster-file', '-cf', type=Path, default=None)
    
    opts = vars(parser.parse_args())
    
    if opts['cluster_file'] is not None:
        conf_files = [opts['cluster_file']]
    else:
        conf_files = [f for f in Path(opts['clusters_files_dir']).iterdir() if f.is_file()]
    
    if not opts['model_output'].exists():
        opts['model_output'].mkdir(parents=True, exist_ok=False)
    
    print('Preparing data...')
    inputs, outputs = load_original_data(opts['data'])
    
    logging.basicConfig(filename=opts['model_output'] / 'train.log', format='%(asctime)s - %(levelname)s - %(message)s')
    
    for f in conf_files:
        with open(f, 'rb') as cf:
            all_runs = load(cf)
            
        out_dir = opts['model_output'] / f.stem
        if not out_dir.exists():
            out_dir.mkdir(parents=True, exist_ok=False)
            
        # test each run
        stats = dict()
        for run_dict in all_runs:
            stats[run_dict['run_id']] = list()
            # train model on each cluster
            for cluster_id, cluster in run_dict['clusters'].items():
                print('____________________________________________________')
                print(f'Running {f.stem} run {run_dict["run_id"]} cluster {cluster_id}')
                try:
                    cluster_inputs, cluster_outputs = join_files_in_cluster(cluster, inputs, outputs)
                    trainX, testX, trainY, testY = train_test_split(cluster_inputs, cluster_outputs, test_size=0.15, random_state=1)
                    
                    # print(trainX.shape, trainY.shape)

                    hypermodel = RegressionHyperModel((trainX.shape[1],))
                    
                    tuner_rs = RandomSearch(
                        hypermodel,
                        objective='mse',
                        seed=42,
                        max_trials=5,
                        executions_per_trial=1,
                        overwrite=True,
                        directory='./tuner_search', 
                    )

                    tuner_rs.search(trainX, trainY, epochs=500, validation_split=0.2, verbose=1,
                                    callbacks=[early_stop])  #epochs=500

                    best_model_r = tuner_rs.get_best_models(num_models=1)[0]
                    loss_r, mse_r = best_model_r.evaluate(testX, testY)
                    print(f"Random Search - {f.stem} : run{run_dict['run_id']} : {cluster_id}")
                    # print(best_model_r.summary())
                    print(f'loss: {loss_r} | mse: {mse_r}')
                    
                    best_model_r.save(out_dir / f'{f.stem}_run{run_dict["run_id"]}_{cluster_id}.h5')
                    
                    # save stats in global file
                    stats[run_dict['run_id']].append({
                        'run': int(run_dict['run_id']),
                        'cluster': int(cluster_id),
                        'loss': float(loss_r),
                        'mse': float(mse_r),
                        'cluster_size': len(cluster),
                    })
                    
                    with open(out_dir / 'stats.json', 'w') as stats_f:
                        json.dump(stats, stats_f, indent=3)
                    
                    # clear models from memory 
                    if K.backend() == 'tensorflow':
                        K.clear_session()
                        gc.collect()
                        print("Cleared session")
                                            
                except Exception as e:
                    logging.error(f'Error in {f.stem}_run{run_dict["run_id"]}_c{cluster_id}', exc_info=True)
                    print(f'Error in {f.stem}_run{run_dict["run_id"]}_c{cluster_id}', e)
                    exit(1)