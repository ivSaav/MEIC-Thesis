import matplotlib.pyplot as plt
from pathlib import Path
import pandas as pd
import numpy as np
import torch

from typing import List, Dict, Tuple

def plot_data_values(data : np.ndarray, title : str,
                     labels : List[str] = ["R [Rsun]", "B [G]", "alpha [deg]"], 
                     scales : Dict[str, str] = {}, **figkwargs):
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
    v2 = data[:, 1280:1920]
    
    fig, axs = plt.subplots(3, 1, **figkwargs)
    fig.subplots_adjust(hspace=0.8)
    fig.suptitle(title)

    for l0,l1,l2 in zip(v0, v1, v2):
        axs[0].plot(l0, linewidth=0.1)
        axs[1].plot(l1, linewidth=0.1)
        axs[2].plot(l2, linewidth=0.1)
    
    # set labels
    for i, label in enumerate(labels):   
        axs[i].set_ylabel(label) 
        axs[i].set_yscale(scales[label] if label in scales else 'log')
    
    plt.tight_layout()
 
    
def plot_epoch(train_vals : np.ndarray, scaler, path : Path, title : str,
               labels : List[str] = ["R [Rsun]", "B [G]", "alpha [deg]"], 
               scales : Dict[str, str] = {'B [G]':'symlog', 'alpha [deg]': 'linear'}):
    # concatenate columns into a single line to apply scaler
    lines = np.array([np.concatenate(val, axis=0) for val in train_vals])
    lines = scaler.inverse_transform(lines)
    
    plot_data_values(lines, title, labels, scales, dpi=200, figsize=(10, 3*3))
    plt.savefig(path, dpi=200)
    plt.close()
    

def plot_anomalies(anomalies : Tuple[str, float], dataloader : torch.utils.data.DataLoader, title : str = "Anomalies"):
    anomaly_files = set([filename for filename, _ in anomalies])
    
    # get anomaly inputs
    anomaly_inputs = []
    for _, (x, files) in enumerate(dataloader):
        anomaly_inputs.extend([inputs.numpy() for inputs, filename in zip(x, files) if filename in anomaly_files])
    
    # concatenate columns into a single line to apply scaler
    lines = np.array([np.concatenate(val, axis=0) for val in anomaly_inputs])
    lines = dataloader.dataset.scaler.inverse_transform(lines)
    plot_data_values(lines, title, ["R [Rsun]", "B [G]", "alpha [deg]"], {'B [G]':'symlog', 'alpha [deg]': 'linear'})
    
def plot_from_files(filenames : List[Path], columns : List[str] = ['R [Rsun]', 'B [G]', 'alpha [deg]'], 
                    scales : Dict[str, str] = {'B [G]':'symlog', 'alpha [deg]': 'linear'}, **figkwargs):
    
    fig, axs = plt.subplots(nrows=len(columns), ncols=1, sharex=True, **figkwargs)
    for idx, ax in enumerate(axs): 
        ax.set_ylabel(columns[idx])
        ax.set_yscale(scales[columns[idx]] if columns[idx] in scales else 'log')
    
    for path in filenames:
        for idx, ax in enumerate(axs):
            df = pd.read_csv(str(path),  skiprows=2, usecols=columns)
            ax.plot(df[columns[idx]],  linewidth=0.5)
    plt.tight_layout()
    