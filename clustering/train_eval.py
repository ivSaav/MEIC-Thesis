from argparse import ArgumentParser
from pathlib import Path
from pickle import load
import json
import gc
import shutil as sh
import logging
import numpy as np

from keras import backend as K
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import QuantileTransformer
from model import RegressionHyperModel
from keras.callbacks import EarlyStopping
from keras_tuner import RandomSearch

from tools.data import load_original_data, join_files_in_cluster, plot_data_values, scale_data
import matplotlib.pyplot as plt
import pandas as pd

early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

input_cols = ['R [Rsun]', 'B [G]', 'alpha [deg]']
output_cols = ['n [cm^-3]', 'v [km/s]', 'T [MK]']


def plot_values(d, model_name, scaler, labels = ["N","V","T"], scales={}, outputs=True, savepath=Path(".")):
    print(model_name, d.shape)
    d = pd.DataFrame(d)
    d.columns = d.columns.astype(str)
    d = scaler.inverse_transform(d)
    
    if outputs:
        ns = d[:, ::3]
        vs = d[:, 1::3]
        ts = d[:, 2::3]
        d = np.concatenate((ns, vs, ts), axis=1)

    fig = plot_data_values(d, model_name, labels, scales, scale="log", figsize=(10,15))
    plt.savefig(savepath, dpi=200)  
    plt.close(fig)

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--data', '-d', type=Path, required=True)
    parser.add_argument('--model-output', '-m', type=Path, required=True)
    parser.add_argument('--cluster-files-dir', '-c', type=str)
    parser.add_argument('--cluster-file', '-cf', type=Path, default=None)
    parser.add_argument('--save-scalers', '-s', action='store_true', default=False)
    parser.add_argument('--ignore-outlier-runs', action="store_true")    
    parser.add_argument('--run-id', '-r', type=int, default=None)
    
    opts = vars(parser.parse_args())
    
    if opts['cluster_file'] is not None:
        conf_files = [opts['cluster_file']]
    else:
        conf_files = [f for f in Path(opts['clusters_files_dir']).iterdir() if f.is_file()]
    
    if not opts['model_output'].exists():
        opts['model_output'].mkdir(parents=True, exist_ok=False)
    
    print('Preparing data...')
    val_files = []
    with open("../data/testing_profiles.txt", "r") as f:
        val_files = f.readlines()
        val_files = [f.split('.')[0] for f in val_files]    
    
    inputs, outputs, valX, valY, scaler_in, scaler_out = load_original_data(opts['data'], opts['save_scalers'], val_files)
    
    print("Inputs shape: ", inputs.shape)
    print("Outputs shape: ", outputs.shape)
    print("Validation inputs shape: ", valX.shape)
    print("Validation outputs shape: ", valY.shape)
    
    logging.basicConfig(filename=opts['model_output'] / 'train.log', format='%(asctime)s - %(levelname)s - %(message)s')
    logging.captureWarnings(True)
    
    for f in conf_files:
        with open(f, 'rb') as cf:
            all_runs = load(cf)
            
        out_dir = opts['model_output'] / f.stem
        if not out_dir.exists():
            out_dir.mkdir(parents=True, exist_ok=False)
            
        # test each run
        stats = dict()
        for run_dict in all_runs:
            # do not train on runs with outliers (values joined into -1)
            if opts['ignore_outlier_runs'] and len(run_dict['small_clusters']) > 0:
                continue
            
            if opts['run_id'] is not None and run_dict['run_id'] != opts['run_id']:
                continue
            
            if any(len(cluster) < 1000 for cluster in run_dict['clusters'].values()):
                print("Cluster size too small, skipping...")
                continue
            
            stats[run_dict['run_id']] = list()
            
            # train model on each cluster
            for cluster_id, cluster in run_dict['clusters'].items():
                print('____________________________________________________')
                print(f'Running {f.stem} run {run_dict["run_id"]} cluster {cluster_id}')
                try:
                    
                    cluster_inputs, cluster_outputs = join_files_in_cluster(cluster, inputs, outputs)
                    trainX, testX, trainY, testY = train_test_split(cluster_inputs, cluster_outputs, test_size=0.15, shuffle=True)

                    hypermodel = RegressionHyperModel((trainX.shape[1],))
                    tuner_rs = RandomSearch(
                        hypermodel,
                        objective='mse',
                        seed=42,
                        max_trials=40,
                        executions_per_trial=1,
                        overwrite=True,
                        directory='./tmp', 
                    )

                    tuner_rs.search(trainX, trainY, epochs=500, validation_split=0.2, verbose=1, callbacks=[early_stop])
                    
                    for idx, model in enumerate(tuner_rs.get_best_models(num_models=10)):
                        loss, mse = model.evaluate(testX, testY)
                        
                        val_cluster = list(set(cluster).intersection(set(val_files)))
                        
                        valX_cluster = valX.loc[valX["filename"].isin(val_cluster)].iloc[:, 1:].values
                        valY_cluster = valY.loc[valY["filename"].isin(val_cluster)].iloc[:, 1:].values
                        val_loss, mse = model.evaluate(valX_cluster, valY_cluster)
                        print(f"Random Search - {f.stem} : run{run_dict['run_id']} : cluster{cluster_id} : top{idx}")
                        print(f'loss: {loss} | mse: {mse}')
                        
                        model_dir = out_dir / f'run{run_dict["run_id"]}_c{cluster_id}'
                        model.save(model_dir / f"top{idx}.h5")
                        
                        predictions = model.predict(cluster_inputs)
                        plot_values(predictions, f"{f.stem}_run{run_dict['run_id']}_c{cluster_id}_top{idx}", 
                                    scaler_out, savepath=model_dir / f"top{idx}.png")
                        
                        predictions = model.predict(valX_cluster)
                        plot_values(predictions, f"{f.stem}_run{run_dict['run_id']}_c{cluster_id}_top{idx}", 
                                    scaler_out, savepath=model_dir / f"top{idx}_val.png")
                        
                        stats[run_dict['run_id']].append({
                            'run': int(run_dict['run_id']),
                            'top': idx,
                            'cluster': int(cluster_id),
                            'loss': float(loss),
                            'val_loss': float(val_loss),
                            'mse': float(mse),
                            'cluster_size': len(cluster),
                            # 'model_params': tuner_rs.get_best_hyperparameters()[0]
                        })
                        
                        with open(model_dir / 'stats.json', 'w') as stats_f:
                            json.dump(stats, stats_f, indent=3)
                            
                        with open(model_dir / f'top{idx}.txt', "w") as text_file:
                            model.summary(print_fn=lambda x: text_file.write(x + '\n'))
                            text_file.write(f"\n\nloss: {loss} \n")
                            text_file.write(f"val loss: {val_loss} \n")
                        
                        del model
                        del predictions
                        plt.close("all")
                        
                    # clear models from memory 
                    if K.backend() == 'tensorflow':
                        del hypermodel
                        del tuner_rs
                        K.clear_session()
                        gc.collect()
                        sh.rmtree('./tmp/', ignore_errors=True)
                        print("Cleared session")
                
                except Exception as e:
                    logging.error(f'Error in {f.stem}_run{run_dict["run_id"]}_c{cluster_id}', exc_info=True)
                    print(f'Error in {f.stem}_run{run_dict["run_id"]}_c{cluster_id}', e)
                    exit(1)