from argparse import ArgumentParser
from pathlib import Path
from pickle import load
import json
import gc
import shutil as sh
import logging

from keras import backend as K
from sklearn.model_selection import train_test_split
from model import RegressionHyperModel
from keras.callbacks import EarlyStopping
from keras_tuner import RandomSearch

from tools.data import load_original_data, join_files_in_cluster

early_stop = EarlyStopping(monitor='val_loss', patience=25, restore_best_weights=True)

input_cols = ['R [Rsun]', 'B [G]', 'alpha [deg]']
output_cols = ['n [cm^-3]', 'v [km/s]', 'T [MK]']

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--data', '-d', type=Path, required=True)
    parser.add_argument('--model-output', '-m', type=Path, required=True)
    parser.add_argument('--cluster-files-dir', '-c', type=str)
    parser.add_argument('--cluster-file', '-cf', type=Path, default=None)
    parser.add_argument('--save-scalers', '-s', action='store_true', default=False)
    parser.add_argument('--ignore-outlier-runs', action="store_true")    
    
    opts = vars(parser.parse_args())
    
    if opts['cluster_file'] is not None:
        conf_files = [opts['cluster_file']]
    else:
        conf_files = [f for f in Path(opts['clusters_files_dir']).iterdir() if f.is_file()]
    
    if not opts['model_output'].exists():
        opts['model_output'].mkdir(parents=True, exist_ok=False)
    
    print('Preparing data...')
    inputs, outputs, _,_ = load_original_data(opts['data'], opts['save_scalers'])
    
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
            # do not train on runs with outliers (values joined into -1)
            if opts['ignore_outlier_runs'] and len(run_dict['small_clusters']) > 0:
                continue
            
            stats[run_dict['run_id']] = list()
            # train model on each cluster
            for cluster_id, cluster in run_dict['clusters'].items():
                print('____________________________________________________')
                print(f'Running {f.stem} run {run_dict["run_id"]} cluster {cluster_id}')
                try:
                    cluster_inputs, cluster_outputs = join_files_in_cluster(cluster, inputs, outputs)
                    trainX, testX, trainY, testY = train_test_split(cluster_inputs, cluster_outputs, test_size=0.15,
                                                                    shuffle=True)
                    
                    hypermodel = RegressionHyperModel((trainX.shape[1],))
                    
                    tuner_rs = RandomSearch(
                        hypermodel,
                        objective='mse',
                        seed=42,
                        max_trials=5,
                        executions_per_trial=1,
                        overwrite=True,
                        directory='./tmp', 
                    )

                    tuner_rs.search(trainX, trainY, epochs=500, validation_split=0.2, verbose=1,
                                    callbacks=[early_stop])

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
                        'model_params': tuner_rs.get_best_hyperparameters()[0]
                    })
                    
                    with open(out_dir / 'stats.json', 'w') as stats_f:
                        json.dump(stats, stats_f, indent=3)
                    
                    # clear models from memory 
                    if K.backend() == 'tensorflow':
                        K.clear_session()
                        gc.collect()
                        sh.rmtree('./tuner_search/', ignore_errors=True)
                        print("Cleared session")
                                            
                except Exception as e:
                    logging.error(f'Error in {f.stem}_run{run_dict["run_id"]}_c{cluster_id}', exc_info=True)
                    print(f'Error in {f.stem}_run{run_dict["run_id"]}_c{cluster_id}', e)
                    exit(1)