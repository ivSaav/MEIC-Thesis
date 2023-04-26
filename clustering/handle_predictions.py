from keras.models import load_model
from pathlib import Path
import pandas as pd
from pickle import load
from argparse import ArgumentParser

from tools.data import load_original_data, join_files_in_cluster, plot_cluster_preds
from tools.viz import plot_predictions
import matplotlib.pyplot as plt

def save_predictions(opts, out_dir, cluster_filenames, all_predictions, val_files):
    """Save predictions in separate files"""
    preds_dir = out_dir / 'data'
    if not preds_dir.exists() and not opts['no_write']:
        preds_dir.mkdir(parents=True)
    
    val_dir = out_dir / 'val'
    val_dir.mkdir(parents=True, exist_ok=True)
    
    compiled_in = pd.read_csv(opts['data_path'] / 'inputs.csv') 
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
    argparser.add_argument('--cluster-file', '-cf', type=str, required=True)
    argparser.add_argument('--run-id', '-r', type=int, required=True)
    argparser.add_argument('--plots', '-p', action='store_true', default=False)
    argparser.add_argument('--no-write', '-sp', action='store_true', default=False)
    argparser.add_argument('--all-data', '-a', action='store_true', default=False)
    opts = vars(argparser.parse_args())
    
    original_in, original_out, _scaler_in, scaler_out = load_original_data(opts['data_path'])
    
    if opts["all_data"]:
        out_dir = opts['hist_path'] / 'predictions' / f"all_data_t{opts['run_id']}"
        clusters = {
            0: list(original_in["filename"].values)
        }
    else:
        out_dir = opts['hist_path'] / 'predictions' / f"{opts['cluster_file']}_{opts['run_id']}"
        clusters_path = opts['hist_path'] / 'clusters'
        cluster_conf = load(open(clusters_path / f"{opts['cluster_file']}.pkl", 'rb'))
        clusters = cluster_conf[opts['run_id']]['clusters']
    
    if not out_dir.exists():
        out_dir.mkdir(parents=True, exist_ok=False)
        
    all_predictions = []
    cluster_filenames = []
    pfs = {}
    for cluster_id, cluster in clusters.items():
        if opts['all_data']:
            model_file = f"{opts['cluster_file']}/top{opts['run_id']}.h5"
        else:
            model_file = f"{opts['cluster_file']}/run{opts['run_id']}_c{cluster_id}/top0.h5"
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
    with open(opts["data_path"].parent / "testing_profiles.txt", 'r') as f:
        val_files = f.readlines()
        val_files = [f.split(".")[0] for f in val_files]
    val_files = set(val_files)
    
    # save predictions in separate files
    if not opts['no_write']:
        print("Saving Predictions...")
        save_predictions(opts, out_dir, cluster_filenames, all_predictions, val_files)
        
        # preds_dir = out_dir / 'data'
        # if not preds_dir.exists() and not opts['no_write']:
        #     preds_dir.mkdir(parents=True)
        
        # val_dir = out_dir / 'val'
        # val_dir.mkdir(parents=True, exist_ok=True)
        
           
        # compiled_in = pd.read_csv(opts['data_path'] / 'inputs.csv') 
        # for f, pred in zip(cluster_filenames, all_predictions):
        #     ns, vs, ts = list(pred[::3]), list(pred[1::3]), list(pred[2::3])            
        #     outputs = pd.DataFrame({'n [cm^-3]': ns, 'v [km/s]': vs, 'T [MK]': ts})
            
        #     inputs = compiled_in.loc[compiled_in['filename'] == f].iloc[:, 1:].values[0]
        #     rs, bs, alphas = list(inputs[:640]), list(inputs[640:1280]), list(inputs[1280:])
        #     inputs = pd.DataFrame({'R [Rsun]': rs, 'B [G]': bs, 'alpha [deg]': alphas})

        #     df = pd.concat([inputs, outputs], axis=1)
            
        #     if f in val_files:
        #         df.to_csv(val_dir / f'{f}.csv', index=False)
        #     else:
        #         df.to_csv(preds_dir / f'{f}.csv', index=False)
    
    # plot predictions
    if opts['plots']:
        print("Plotting Predictions...")
        fig = plot_predictions(cluster_filenames, all_predictions, "Predictions", opts['data_path'] / "outputs.csv")
        plt.savefig(out_dir / 'predictions.png', dpi=200)
        plt.close(fig)
        
        fig = plot_predictions(cluster_filenames, all_predictions, "Predictions - Validation", opts['data_path'] / "outputs.csv", val_files)
        plt.savefig(out_dir / 'predictions_val.png', dpi=200)
        
        exit()
        fig, axs = plt.subplots(3, 2, figsize=(20, 15), dpi=200, sharey="row" ,sharex="all")
        real_out = pd.read_csv(opts['data_path'] / 'outputs.csv')
        for f, pred in zip(cluster_filenames, all_predictions):
            ns, vs, ts = list(pred[::3]), list(pred[1::3]), list(pred[2::3])
            
            real = real_out.loc[real_out['filename'] == f].iloc[:, 1:].values[0]
            real_ns, real_vs, real_ts = list(real[:640]), list(real[640:1280]), list(real[1280:])
            
            axs[0,0].plot(real_ns, linewidth=0.5)
            axs[0,1].plot(ns, linewidth=0.5)
            
            axs[1,0].plot(real_vs, linewidth=0.5)
            axs[1,1].plot(vs, linewidth=0.5)
            
            axs[2,0].plot(real_ts, linewidth=0.5)
            axs[2,1].plot(ts, linewidth=0.5)
        
        for ax in axs.flat:
            ax.set_yscale('log')
            
        axs[0,0].set_ylabel('n [cm^-3]')
        axs[1,0].set_ylabel('v [km/s]')
        axs[2,0].set_ylabel('T [MK]')
        
        axs[0,0].set_title('Real')
        axs[0,1].set_title('Predicted')
        
        plt.tight_layout()
        
        plt.savefig(out_dir / 'predictions.png')
            
    