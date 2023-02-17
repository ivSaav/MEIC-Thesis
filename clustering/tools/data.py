from pickle import dump
from sklearn.preprocessing import QuantileTransformer
import pandas as pd
from typing import Tuple, List
from pathlib import Path

import matplotlib.pyplot as plt
import seaborn as sns

def load_original_data(data_path: Path, save_scalers : bool = False) -> Tuple[pd.DataFrame, pd.DataFrame, QuantileTransformer, QuantileTransformer]:
    """Load the original data from the file."""

    inputs = pd.read_csv(data_path / 'inputsdata_compilation.csv')
    outputs = pd.read_csv(data_path / 'outputsdata_compilation.csv')
    
    input_filenames = inputs[['filename']]
    output_filenames = outputs[['filename']]
    
    scaler_inputs, scaler_ouputs = QuantileTransformer(), QuantileTransformer()
    inputs = scaler_inputs.fit_transform(inputs.iloc[:, 1:])
    outputs = scaler_ouputs.fit_transform(outputs.iloc[:, 1:])
    
    if save_scalers:
        dump((scaler_inputs, scaler_ouputs), open(data_path / 'scalers.pkl', 'wb'))
    
    inputs = pd.DataFrame(inputs)
    inputs = pd.concat([input_filenames, inputs], axis=1)
    
    outputs = pd.DataFrame(outputs)
    outputs = pd.concat([output_filenames, outputs], axis=1)
    
    print("Scaled inputs:", inputs.head())
    print("Scaled outputs:", outputs.head())
    return inputs, outputs, scaler_inputs, scaler_ouputs


def join_files_in_cluster(cluster_files: List[Path], input_data : pd.DataFrame, output_data : pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Join all files in a cluster into a single dataframe."""
    cluster_inputs, cluster_outputs = pd.DataFrame(), pd.DataFrame()
    
    inputs = [input_data.loc[input_data['filename'] == f].iloc[:, 1:]
              for f in cluster_files]
    
    cluster_inputs = pd.concat(inputs, axis=0, ignore_index=True)
    
    outputs = [output_data.loc[output_data['filename'] == f].iloc[:, 1:]
               for f in cluster_files]
    cluster_outputs = pd.concat(outputs, axis=0, ignore_index=True)      
    
    # print(cluster_inputs.head())
    # print(cluster_inputs.shape)
    # print(cluster_df)
    # print(cluster_df.shape)
    # print(cluster_df.columns)
    print("Cluster shape:", cluster_inputs.shape)
    return cluster_inputs, cluster_outputs


def plot_cluster_preds(pred_df : pd.DataFrame, model_name : str, out_dir : Path):
    """Plot the predictions of a cluster."""
    ns = pred_df.iloc[:, 0:640]
    vs = pred_df.iloc[:, 640:1280]
    ts = pred_df.iloc[:, 1280:1920]
    
    print(ns.shape)
    print(ns.values)
    
    print(vs.head())
    print(ts.head())
    
    vs.columns = [i for i in range(640)]
    ts.columns = [i for i in range(640)]
    
    print(vs.head())
    
    
    fig, axs = plt.subplots(3, 1, figsize=(12, 6*3))
    fig.suptitle(f'Predictions for {model_name}')


    for _, n in ns.iterrows():
        axs[0].plot(n, linewidth=0.1)
    for _, v in vs.iterrows():
        axs[1].plot(v, linewidth=0.1)
    for _, t in ts.iterrows():
        axs[2].plot(t, linewidth=0.1)
        
    axs[0].set_ylabel('n (m^-3)')
    axs[1].set_ylabel('v (m/s)')
    axs[2].set_ylabel('T (MK)')    
   
    for ax in axs:
        ax.set_yscale('log')
    plt.tight_layout()
    plt.savefig(out_dir / f'{model_name}.png', dpi=500)  
    