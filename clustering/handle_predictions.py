from keras.models import load_model
from pathlib import Path
import pandas as pd
from pickle import load
from argparse import ArgumentParser

from tools.data import load_original_data, join_files_in_cluster

import matplotlib.pyplot as plt

if __name__ == '__main__':
    argparser = ArgumentParser()
    argparser.add_argument('--data_path', '-d', type=Path, required=True)
    argparser.add_argument('--hist-path', '-hp', type=Path, required=True)
    argparser.add_argument('--cluster-file', '-cf', type=str, required=True)
    argparser.add_argument('--run-id', '-r', type=int, required=True)
    opts = vars(argparser.parse_args())
    
    out_dir = opts['hist_path'] / 'predictions' / f"{opts['cluster_file']}_{opts['run_id']}"
    if not out_dir.exists():
        out_dir.mkdir(parents=True, exist_ok=False)
        
    clusters_path = opts['hist_path'] / 'clusters'
    
    original_in, original_out, scaler_in, scaler_out = load_original_data(opts['data_path'])
    
    cluster_conf = load(open(clusters_path / f"{opts['cluster_file']}.pkl", 'rb'))
    clusters = cluster_conf[opts['run_id']]['clusters']
    
    all_predictions = []
    cluster_filenames = []
    for cluster_id, cluster in clusters.items():
        model_file = f"{opts['cluster_file']}_run{opts['run_id']}_{cluster_id}.h5"
        model = load_model(opts['hist_path'] / f"models/{opts['cluster_file']}" / model_file)
        
        # print(cluster_id)
        # print(cluster)
        inputs, _outputs = join_files_in_cluster(cluster, original_in, original_out)
        
        predictions = model.predict(inputs)
        print(predictions.shape)
        
        cluster_filenames.append(cluster)
        all_predictions.append(pd.DataFrame(predictions))
    
    # do inverse tranform on the predictions
    all_predictions = pd.concat(all_predictions, ignore_index=True)
    all_predictions.columns = all_predictions.columns.astype(str)
    all_predictions = scaler_out.inverse_transform(all_predictions)
    all_predictions = pd.DataFrame(all_predictions)
    
    
    ns = original_out.iloc[:, 1:641]
    
    fig, axs = plt.subplots()
    for line in ns.values:
        axs.plot(line, linewidth=0.1)
        break
    axs.set_yscale('log')
    plt.savefig(out_dir / 'ns.png')
    print(ns.values)
    # for i in range(all_predictions.shape[0]):
        
    print(all_predictions.shape)
    print(all_predictions.head())
    
    
    
    
    
    