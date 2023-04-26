import matplotlib.pyplot as plt
import pandas as pd
from typing import List, Set
import numpy as np
from pathlib import Path

def plot_predictions(filenames : List[str], predictions : np.ndarray, title : str, 
                     real_path : Path, val_files : Set[str] = []) -> plt.figure:
    fig, axs = plt.subplots(3, 2, figsize=(20, 15), dpi=200, sharey="row" ,sharex="all")
    real_out = pd.read_csv(real_path)
    for f, pred in zip(filenames, predictions):
        if len(val_files) > 0 and f not in val_files: continue
        
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
    
    fig.suptitle(title, fontsize=16)
    plt.tight_layout()
    return fig