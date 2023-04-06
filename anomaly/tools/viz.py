import matplotlib.pyplot as plt
from pathlib import Path
import pandas as pd
import numpy as np
import io

import PIL.Image
from torchvision.transforms import ToTensor

from typing import List, Dict, Tuple

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
    if "R [Rsun]" in labels:
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


def plot_single_var(values : np.ndarray, title : str,
                       label : str = "B [G]", scale : str = 'log', **figkwargs):
    fig, ax = plt.subplots(**figkwargs)
    for line in values:
        ax.plot(line, linewidth=0.1)
    ax.set_ylabel(label)
    ax.set_yscale(scale)
    plt.title(title)
    plt.tight_layout()
    return fig
   
 
    
def plot_epoch(train_vals : np.ndarray, scaler, path : Path, title : str,
               labels : List[str] = ["R [Rsun]", "B [G]", "alpha [deg]"], 
               scales : Dict[str, str] = {'B [G]':'log', 'alpha [deg]': 'linear'}):
    # concatenate columns into a single line to apply scaler
    lines = np.array([np.concatenate(val, axis=0) for val in train_vals])
    lines = scaler.inverse_transform(lines)
    
    plot_data_values(lines, title, labels, scales, dpi=200, figsize=(10, 3*3))
    plt.savefig(path, dpi=200)
    plt.close()
    

    

def plot_anomalies(anomalies : Tuple[str, float], data_path : Path, title : str = "Anomalies", inverse=False, **figkwargs):
    # get all compiled inputs
    df = pd. read_csv(data_path)
    
    if inverse:
        df = df[~df["filename"].isin(anomalies)].iloc[:, 1:]
    else:
        # select the rows with the filenames in anomalies
        df = df[df["filename"].isin(anomalies)].iloc[:, 1:]
    return plot_data_values(df.values, title, scales={'B [G]':'log', 'alpha [deg]': 'symlog'}, **figkwargs)
    
def plot_from_files(filenames : List[Path], columns : List[str] = ['R [Rsun]', 'B [G]', 'alpha [deg]'], 
                    scales : Dict[str, str] = {'B [G]':'log', 'alpha [deg]': 'linear'}, **figkwargs):
    
    fig, axs = plt.subplots(nrows=len(columns), ncols=1, sharex=True, **figkwargs)
    for idx, ax in enumerate(axs): 
        ax.set_ylabel(columns[idx])
        ax.set_yscale(scales[columns[idx]] if columns[idx] in scales else 'log')
    
    for path in filenames:
        for idx, ax in enumerate(axs):
            df = pd.read_csv(str(path), usecols=columns)
            ax.plot(df[columns[idx]],  linewidth=0.5)
    plt.tight_layout()
    
    
def plot_to_tensorboard(writer, fig, step, tag="train_plots"):
    """
    Takes a matplotlib figure handle and converts it using
    canvas and string-casts to a numpy array that can be
    visualized in TensorBoard using the add_image function
    Parameters:
        writer (tensorboard.SummaryWriter): TensorBoard SummaryWriter instance.
        fig (matplotlib.pyplot.fig): Matplotlib figure handle.
        step (int): counter usually specifying steps/epochs/time.
    """

    buf = io.BytesIO()
    fig.savefig(buf, format='jpeg')
    buf.seek(0)
    
    image = PIL.Image.open(buf)
    image = ToTensor()(image)
    
    writer.add_image(tag, image, step)
    return fig
    # plt.close(fig)
    
    
def plot_anomaly_scores(scores : List[Tuple[str, float]], percent : float, data_path : Path, save_path : Path = None, 
                logger = None, logger_var : str ="test/", scale="linear", method : str = "", 
                normal_plot : bool =False, exclude : List[str] = []):
    # calculate anomaly threshold based on percentage of anomalies
    sorted_scores = sorted([s[1] for s in scores], reverse=True)
    t = sorted_scores[int(len(sorted_scores)*percent)]
    print("Anomaly Threshold: ", t)
    
    # plot anomaly scores with calculated threshold
    scores_fig, ax = plt.subplots(figsize=(20, 5))
    ax.plot([score[1] for score in scores], label='Anomaly Score', linewidth=0.3)
    ax.plot(t*np.ones(len(scores)), label=f'Threshold ({int(t)})')
    ax.set_yscale(scale)
    plt.legend()
    
    anomalies = [score[0] for score in scores if score[1] > t]
    print(f"Found {len(anomalies)} anomalies")
    anomal_fig = plot_anomalies(anomalies, data_path, f"{method + ' '}Anomalies - {len(anomalies)}", figsize=(8, 5), dpi=200)
    if normal_plot:
        anomalies.extend(exclude) # exclude files (for profile filtering)
        data_fig = plot_anomalies(anomalies, data_path, "Dataset", inverse=True, figsize=(8, 5), dpi=200)
    
    if save_path != None: 
        scores_fig.savefig(str(save_path) + "_scores", dpi=200)
        anomal_fig.savefig(str(save_path) + "_anomalies", dpi=200)
        if normal_plot: data_fig.savefig(str(save_path) + "_normal", dpi=200)
        
    if logger != None: 
        plot_to_tensorboard(logger, scores_fig, 0, logger_var)
        plot_to_tensorboard(logger, anomal_fig, 0, logger_var + "_anomalies")
    
    return t, scores_fig

    


def plot_train_hist(D_losses, G_losses, out_dir : Path = None):
    fig = plt.figure(figsize=(10,5))
    plt.title("Generator and Discriminator Loss During Training")
    plt.plot(G_losses,label="G")
    plt.plot(D_losses,label="D")
    plt.xlabel("iterations")
    plt.ylabel("Loss")
    plt.legend()
    
    if out_dir: fig.savefig(out_dir / "img/train_hist", dpi=200)
    return fig

