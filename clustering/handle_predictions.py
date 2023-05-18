from keras.models import load_model
from pathlib import Path
import pathlib
import pandas as pd
from pickle import load
from argparse import ArgumentParser

from tools.data import load_original_data, join_files_in_cluster, plot_cluster_preds
from tools.viz import plot_predictions
import matplotlib.pyplot as plt

def save_predictions(opts, out_dir, cluster_filenames, all_predictions, val_files, in_name = "inputs"):
    """Save predictions in separate files"""
    preds_dir = out_dir / 'data'
    if not preds_dir.exists() and not opts['no_write']:
        preds_dir.mkdir(parents=True)

    if len(val_files) > 0:
        val_dir = out_dir / 'val'
        val_dir.mkdir(parents=True, exist_ok=True)
    
    compiled_in = pd.read_csv(opts['data_path'] / f'{in_name}.csv') 
    for f, pred in zip(cluster_filenames, all_predictions):
        ns, vs, ts = list(pred[::3]), list(pred[1::3]), list(pred[2::3])            
        outputs = pd.DataFrame({'n [cm^-3]': ns, 'v [km/s]': vs, 'T [MK]': ts})
        
        inputs = compiled_in.loc[compiled_in['filename'] == f].iloc[:, 1:].values[0]
        rs, bs, alphas = list(inputs[:640]), list(inputs[640:1280]), list(inputs[1280:])
        inputs = pd.DataFrame({'R [Rsun]': rs, 'B [G]': bs, 'alpha [deg]': alphas})

        df = pd.concat([inputs, outputs], axis=1)
        if f in val_files:
            df.to_csv(val_dir / f'{f}.csv', index=False)
        else:
            df.to_csv(preds_dir / f'{f}.csv', index=False)

if __name__ == '__main__':
    argparser = ArgumentParser()
    argparser.add_argument('--data_path', '-d', type=Path, required=True)
    argparser.add_argument('--hist-path', '-hp', type=Path, required=True)
    argparser.add_argument('--model-file', '-mf', type=str, required=True)
    argparser.add_argument('--cluster-file', '-cf', type=str, required=True)
    argparser.add_argument('--run-id', '-r', type=int, required=True)
    argparser.add_argument('--plots', '-p', action='store_true', default=False)
    argparser.add_argument('--no-write', '-sp', action='store_true', default=False)
    argparser.add_argument('--all-data', '-a', action='store_true', default=False)
    argparser.add_argument('--top', '-t', type=int, default=0)
    argparser.add_argument('--val', '-v', action='store_true', default=False)
    opts = vars(argparser.parse_args())
    
    original_in, original_out, _scaler_in, scaler_out =\
          load_original_data(opts['data_path'],
                             in_name="inputs" if not opts['val'] else "inputs_val",
                             out_name="outputs_inter" if not opts['val'] else "outputs_inter_val")
    
    print(original_in.shape, original_out.shape)
    
    if opts["all_data"]:
        out_dir = opts['hist_path'] / 'predictions' / f"all_data_t{opts['run_id']}"
        clusters = {
            0: list(original_in["filename"].values)
        }
    else:
        out_dir = opts['hist_path'] / 'predictions' / f"{opts['cluster_file']}_{opts['run_id']}_t{opts['top']}"
        clusters_path = opts['hist_path'] / 'clusters'
        clusters_path = Path(clusters_path / f"{opts['cluster_file']}.pkl")
        cluster_conf = load(open(clusters_path, 'rb'))
        clusters = cluster_conf[opts['run_id']]['clusters']
    
    print(clusters.keys(), type(clusters), type(clusters[0]))
    if not out_dir.exists():
        out_dir.mkdir(parents=True, exist_ok=False)
    
    # exit()
    all_predictions = []
    cluster_filenames = []
    for cluster_id, cluster in clusters.items():
        # print(cluster)
        if opts['all_data']:
            model_file = f"{opts['model_file']}/top{opts['top']}.h5"
        else:
            model_file = f"{opts['model_file']}/run{opts['run_id']}_c{cluster_id}/top{opts['top']}.h5"
        model = load_model(opts['hist_path'] / "models" / model_file)
        
        inputs, outputs = join_files_in_cluster(cluster, original_in, original_out)
        
        predictions = model.predict(inputs)
        cluster_filenames.extend(cluster)
        all_predictions.append(pd.DataFrame(predictions))
    all_predictions = pd.concat(all_predictions, ignore_index=True)
    
    # do inverse tranform on the predictions
    all_predictions.columns = all_predictions.columns.astype(str)
    all_predictions = scaler_out.inverse_transform(all_predictions)
    
    print("Preds shape: ", all_predictions.shape)
    print(all_predictions)
  
    # load validation filenames
    val_files = [] 
    if opts['val']:
        with open(opts["data_path"].parent / "testing_profiles.txt", 'r') as f:
            val_files = f.readlines()
            val_files = [f.split(".")[0] for f in val_files]
        val_files = set(val_files)
    
    # save predictions in separate files
    if not opts['no_write']:
        print("Saving Predictions...")
        save_predictions(opts, out_dir, cluster_filenames, all_predictions, val_files, in_name="inputs" if not opts['val'] else "inputs_val")
    
    # plot predictions
    if opts['plots']:
        print("Plotting Predictions...")
        if not opts["val"]:
            fig = plot_predictions(cluster_filenames, all_predictions, "Predictions", opts['data_path'] / "outputs.csv")
            plt.savefig(out_dir / 'predictions.png', dpi=200)
            plt.close(fig)
        else:
            fig = plot_predictions(cluster_filenames, all_predictions, "Predictions - Validation", opts['data_path'] / "outputs_val.csv", val_files)
            plt.savefig(out_dir / 'predictions_val.png', dpi=200)
        
    