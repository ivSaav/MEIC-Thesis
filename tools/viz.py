import matplotlib.pyplot as plt
from pathlib import Path
import pandas as pd

from typing import List, Dict

def plot_values(values_df : pd.DataFrame, labels : List[str] = ["R [Rsun]", "B [G]", "alpha [deg]"], scales : Dict[str, str] = {}):
    """Plot the predictions of a cluster."""
    v0 = values_df.iloc[:, 0:640]
    v1 = values_df.iloc[:, 640:1280]
    v2 = values_df.iloc[:, 1280:1920]
    
    v1.columns = [i for i in range(640)]
    v2.columns = [i for i in range(640)]
    
    fig, axs = plt.subplots(3, 1, figsize=(10, 3*3))
    fig.subplots_adjust(hspace=0.8)
    # fig.suptitle(f'')

    for _, n in v0.iterrows():
        axs[0].plot(n, linewidth=0.1)
    for _, v in v1.iterrows():
        axs[1].plot(v, linewidth=0.1)
    for _, t in v2.iterrows():
        axs[2].plot(t, linewidth=0.1)
    
    # set labels
    for i, label in enumerate(labels):   
        axs[i].set_ylabel(label) 
        axs[i].set_yscale(scales[label] if label in scales else 'log')
    
    plt.tight_layout()
    plt.show()