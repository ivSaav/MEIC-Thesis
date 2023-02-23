from keras.models import load_model
from pathlib import Path
import pandas as pd
from pickle import load
from argparse import ArgumentParser

from tools.data import load_original_data, join_files_in_cluster, plot_cluster_preds
from sklearn.metrics import mean_squared_error

if __name__ == '__main__':
    argparser = ArgumentParser()
    argparser.add_argument('--data_path', '-d', type=Path, required=True)
    argparser.add_argument('--hist-path', '-hp', type=Path, required=True)
    argparser.add_argument('--cluster-file', '-cf', type=str, required=True)
    argparser.add_argument('--run-id', '-r', type=int, required=True)
    argparser.add_argument('--plots', '-p', action='store_true', default=False)
    argparser.add_argument('--no-write', '-sp', action='store_true', default=False)
    opts = vars(argparser.parse_args())
    
    out_dir = opts['hist_path'] / 'predictions' / f"{opts['cluster_file']}_{opts['run_id']}"
    if not out_dir.exists() and not opts['no_write']:
        out_dir.mkdir(parents=True, exist_ok=False)
        
    clusters_path = opts['hist_path'] / 'clusters'
    
    original_in, original_out, _scaler_in, scaler_out = load_original_data(opts['data_path'])
    
    cluster_conf = load(open(clusters_path / f"{opts['cluster_file']}.pkl", 'rb'))
    clusters = cluster_conf[opts['run_id']]['clusters']
    
    all_predictions = []
    cluster_filenames = []
    for cluster_id, cluster in clusters.items():
        model_file = f"{opts['cluster_file']}_run{opts['run_id']}_{cluster_id}.h5"
        model = load_model(opts['hist_path'] / f"models/{opts['cluster_file']}" / model_file)
        
        inputs, _outputs = join_files_in_cluster(cluster, original_in, original_out)
        
        predictions = model.predict(inputs)
        print(predictions.shape)
        
        cluster_filenames.extend(cluster)
        all_predictions.append(pd.DataFrame(predictions))
    
    all_predictions = pd.concat(all_predictions, ignore_index=True)
    
    # mse = mean_squared_error(original_out.iloc[:, 1:], all_predictions)
    
    # do inverse tranform on the predictions
    all_predictions.columns = all_predictions.columns.astype(str)
    all_predictions = scaler_out.inverse_transform(all_predictions)
    
    
    print(all_predictions.shape)
    print(all_predictions)
    preds_dir = out_dir / 'data'
    if not preds_dir.exists() and not opts['no_write']:
        preds_dir.mkdir(parents=True)
    
    compiled_in = pd.read_csv(opts['data_path'] / 'inputsdata_compilation.csv') 
    for f, pred in zip(cluster_filenames, all_predictions):
        ns, vs, ts = list(pred[:640]), list(pred[640:1280]), list(pred[1280:])
        outputs = pd.DataFrame({'n [cm^-3]': ns, 'v [km/s]': vs, 'T [MK]': ts})
        
        inputs = compiled_in.loc[compiled_in['filename'] == f].iloc[:, 1:].values[0]
        rs, bs, alphas = list(inputs[:640]), list(inputs[640:1280]), list(inputs[1280:])
        inputs = pd.DataFrame({'R [Rsun]': rs, 'B [G]': bs, 'alpha [deg]': alphas})

        df = pd.concat([inputs, outputs], axis=1)
        df.to_csv(preds_dir / f'{f}.csv', index=False)
        
        
    
    # all_predictions = pd.DataFrame(all_predictions)
    # if opts['plots']:
    #     plot_cluster_preds(all_predictions, opts['cluster_file'], out_dir)
    # all_predictions = pd.concat([pd.DataFrame(cluster_filenames, columns=["filename"]), all_predictions], axis=1)
    # if not opts['no_write']:    
    #     print(all_predictions.head())
    #     all_predictions.to_csv(out_dir / f"predictions_compiled.csv", index=False)
    
    # # print(original_out.shape, all_predictions.shape)
    # # print("MSE: ", mse)
    
    # with open('metrics.txt', 'a') as f:
    #     f.write(f"{opts['cluster_file']} : {opts['run_id']} : {mse}\n")
        
    
    
    
    
    