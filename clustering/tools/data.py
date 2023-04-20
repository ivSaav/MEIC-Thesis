from pickle import dump
from sklearn.preprocessing import QuantileTransformer
from sklearn.utils import shuffle
import pandas as pd
from typing import Tuple, List, Dict
from pathlib import Path
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns
# sns.set()

def load_original_data(data_path: Path, save_scalers : bool = False, 
                       val_files : List[str] = []) -> Tuple[pd.DataFrame, pd.DataFrame, QuantileTransformer, QuantileTransformer]:
    """Load the original data from the file."""

    inputs = pd.read_csv(data_path / 'inputs.csv')
    outputs = pd.read_csv(data_path / 'outputs_inter.csv')
    
    inputs, outputs = shuffle(inputs, outputs, random_state=1)

    val_inputs = [inputs.loc[inputs['filename'] == f]
                  for f in val_files]
    val_outputs = [outputs.loc[outputs['filename'] == f]
                   for f in val_files]
    
    inputs = [inputs.loc[inputs['filename'] == f]
              for f in inputs["filename"].values if f not in val_files]
    
    outputs = [outputs.loc[outputs['filename'] == f]
               for f in outputs["filename"].values if f not in val_files]
    
    val_inputs = pd.concat(val_inputs, axis=0, ignore_index=True)
    val_outputs = pd.concat(val_outputs, axis=0, ignore_index=True)
    inputs = pd.concat(inputs, axis=0, ignore_index=True)
    outputs = pd.concat(outputs, axis=0, ignore_index=True)
    
    input_filenames = inputs[['filename']]
    output_filenames = outputs[['filename']]
    scaler_inputs = QuantileTransformer()
    scaler_outputs = QuantileTransformer()
    inputs = scaler_inputs.fit_transform(inputs.iloc[:, 1:])
    outputs = scaler_outputs.fit_transform(outputs.iloc[:, 1:])
    
    val_input_filenames = val_inputs[['filename']]
    val_output_filenames = val_outputs[['filename']]
    val_inputs = scaler_inputs.transform(val_inputs.iloc[:, 1:])
    val_outputs = scaler_outputs.transform(val_outputs.iloc[:, 1:])
    
    inputs = pd.DataFrame(inputs)
    inputs = pd.concat([input_filenames, inputs], axis=1)
    val_inputs = pd.DataFrame(val_inputs)
    val_inputs = pd.concat([val_input_filenames, val_inputs], axis=1)
    
    outputs = pd.DataFrame(outputs)
    outputs = pd.concat([output_filenames, outputs], axis=1)
    val_outputs = pd.DataFrame(val_outputs)
    val_outputs = pd.concat([val_output_filenames, val_outputs], axis=1)
    
    if len(val_files) > 0:
        return inputs, outputs, val_inputs, val_outputs, scaler_inputs, scaler_outputs
    return inputs, outputs, scaler_inputs, scaler_outputs
    
    
    input_val, output_val = None, None
    if val_files:
        with open(val_files, "r") as f:
            val_filenames = f.readlines()
            val_filenames = set([x.split(".")[0] for x in val_filenames])

            # extract validation files
            input_val = inputs.loc[inputs['filename'].isin(val_filenames)].reset_index(drop=True)
            output_val = outputs.loc[outputs['filename'].isin(val_filenames)].reset_index(drop=True)
            # exclude validation files from training
            inputs = inputs.loc[~inputs['filename'].isin(val_filenames)].reset_index(drop=True)
            outputs = outputs.loc[~outputs['filename'].isin(val_filenames)].reset_index(drop=True)
    
    # save filenames before tranforming
    input_filenames = inputs[['filename']]
    output_filenames = outputs[['filename']]
    # transform inputs and outputs (without validation files)
    scaler_inputs, scaler_ouputs = QuantileTransformer(), QuantileTransformer()
    inputs = scaler_inputs.fit_transform(inputs.iloc[:, 1:])
    outputs = scaler_ouputs.fit_transform(outputs.iloc[:, 1:])
    
    if save_scalers:
        dump((scaler_inputs, scaler_ouputs), open(data_path / 'scalers.pkl', 'wb'))
    
    # transform validation files
    if val_files:
        input_val_filenames = input_val[['filename']]
        output_val_filenames = output_val[['filename']]
        input_val = scaler_inputs.transform(input_val.iloc[:, 1:])
        output_val = scaler_ouputs.transform(output_val.iloc[:, 1:])
        
        input_val = pd.DataFrame(input_val)
        input_val = pd.concat([input_val_filenames, input_val], axis=1)
        output_val = pd.DataFrame(output_val)
        output_val = pd.concat([output_val_filenames, output_val], axis=1)

    inputs = pd.DataFrame(inputs)
    inputs = pd.concat([input_filenames, inputs], axis=1)
    
    outputs = pd.DataFrame(outputs)
    outputs = pd.concat([output_filenames, outputs], axis=1)
        
    print("Scaled inputs:", inputs.head())
    print("Scaled outputs:", outputs.head())
    
    if val_files:
        return inputs, outputs, input_val , output_val, scaler_inputs, scaler_ouputs
    return inputs, outputs, scaler_inputs, scaler_ouputs

def scale_data(data : pd.DataFrame, scaler : QuantileTransformer) -> pd.DataFrame:
    """Scale the data using the given scaler."""
    scaled = scaler.transform(data.iloc[:, 1:])
    scaled = pd.DataFrame(scaled)
    data = pd.concat([data.iloc[:, 0], scaled], axis=1)
    return data


def join_files_in_cluster(cluster_files: List[Path], input_data : pd.DataFrame, output_data : pd.DataFrame,
                          val_files : List[str] = []) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Join all files in a cluster into a single dataframe."""
    cluster_inputs, cluster_outputs = pd.DataFrame(), pd.DataFrame()
    
    # exclude validation files in training
    cluster_files = set(cluster_files) - set(val_files)
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
    
    vs.columns = [i for i in range(640)]
    ts.columns = [i for i in range(640)]
    
    fig, axs = plt.subplots(3, 1, figsize=(12, 6*3))
    fig.subplots_adjust(hspace=0.8)
    fig.suptitle(f'Predictions for {model_name}')

    for _, n in ns.iterrows():
        axs[0].plot(n, linewidth=0.5)
    for _, v in vs.iterrows():
        axs[1].plot(v, linewidth=0.5)
    for _, t in ts.iterrows():
        axs[2].plot(t, linewidth=0.5)
        
    axs[0].set_ylabel('n (m^-3)')
    axs[0].set_yscale("log")
    axs[1].set_ylabel('v (m/s)')
    axs[2].set_ylabel('T (MK)')    
   
    # for ax in axs:
    #     ax.set_yscale('linear')
    plt.tight_layout()
    plt.savefig(out_dir / f'{model_name}.png', dpi=500)  
 
    
def plot_data_values(data : np.ndarray, title : str,
                     labels : List[str] = ["R [Rsun]", "B [G]", "alpha [deg]"], 
                     scales : Dict[str, str] = {}, scale : str ="log", **figkwargs):
    """
    Plot 3 data columns at once.
    Args:
        data (np.ndarray): np array of shape (n, 1920)
        title (str): plot title
        labels (List[str], optional): ylabels. Defaults to ["R [Rsun]", "B [G]", "alpha [deg]"].
        scales (Dict[str, str], optional): yscale dictionary. Defaults to {}.
    """    
    v0 = data[:, 0:640]
    v1 = data[:, 640:1280]
    v2 = []
    if "R [Rsun]" or "N" in labels:
        v2 = data[:, 1280:1920]    
    
    fig, axs = plt.subplots(len(labels), 1, **figkwargs)
    fig.subplots_adjust(hspace=0.8)
    fig.suptitle(title)
    
    for l0,l1 in zip(v0, v1):
        axs[0].plot(l0, linewidth=0.1)
        axs[1].plot(l1, linewidth=0.1)
        
    if len(labels) > 2:
        for l2 in v2:
            axs[2].plot(l2, linewidth=0.1)
    
    # set labels
    for i, label in enumerate(labels):   
        axs[i].set_ylabel(label) 
        axs[i].set_yscale(scales[label] if label in scales else scale)
        
    plt.tight_layout()
    return fig
    