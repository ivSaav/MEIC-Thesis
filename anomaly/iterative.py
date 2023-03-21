from multivp_dataset import MULTI_VP_Dataset
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from sklearn.preprocessing import StandardScaler, MinMaxScaler
import datetime
from typing import List
import matplotlib.pyplot as plt

from models.aae import Encoder, Decoder, Discriminator, train, reconstruction_anomaly
from tools.viz import plot_anomaly_scores, plot_train_hist
from tools.logging import Logger

time_id = datetime.datetime.now().strftime("%m%d-%H%M")

class Options:
    def __init__(self):
        # Dataset
        self.data_path = Path('../data/compiled/inputs.csv')
        self.batch_size = 128
        self.nworkers = 4
        self.shuffle = True
        self.method = "joint"
        self.scaler = MinMaxScaler((-1, 1))
        self.l_dim = 100
        # self.wsize = 5
        
        # Train params
        self.lr = 0.0002
        self.epochs = 100
        self.sample_interval = self.epochs // 2
        self.train_plots = True
        self.iters = 10
        self.anomaly_threshold = 0.02
        
        self.model_out = Path('./runs/iterative/' + time_id)

        # create ouput dirs
        if not self.model_out.exists(): self.model_out.mkdir(parents=True)
        (self.model_out / "img").mkdir(exist_ok=True)
        
        # logging
        self.tags = ["iterative", "aae", "joint", "minmax", "test"]
        self.desc = "No model reset"
        self.type = "Iter"
        
opts = Options()


def setup_train_dataset(anomalies : List[str], i):
    dataset = MULTI_VP_Dataset(
        path=opts.data_path,
        method=opts.method,
        remove_extreme=True, # TODO try only removing in the first iteration
        is_train=True,
        scaler=opts.scaler, 
    )
    dataset.remove_files(anomalies)
    dataset.plot(f"Data Iter {i}")
    plt.savefig(opts.model_out / f"img/data_iter{i}.png", dpi=200)
    plt.close()
    return DataLoader(dataset, batch_size=opts.batch_size, shuffle=True, num_workers=opts.nworkers, drop_last=False)

def setup_test_dataset(train_dataset : MULTI_VP_Dataset, anomalies : List[str]):
    dataset = MULTI_VP_Dataset(
        path=opts.data_path,
        method=opts.method,
        remove_extreme=False,
        is_train=False,
        scaler=train_dataset.scaler
    )
    dataset.remove_files(anomalies)
    return DataLoader(dataset, batch_size=opts.batch_size, shuffle=False, num_workers=opts.nworkers, drop_last=False)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
anomalies = []


netEnc = Encoder(input_size=1280, l_dim=opts.l_dim, device=device).to(device)
netDec = Decoder(output_size=1280,l_dim=opts.l_dim).to(device)
netD = Discriminator(l_dim=opts.l_dim).to(device)

logger = Logger([netEnc, netDec, netD], opts.model_out / f"{time_id}.md", vars(opts))

for i in range(opts.iters):
    print(f"\n_________ Iteration {i} __________")
    train_dataloader = setup_train_dataset(anomalies, i)
    logger.log(f"## Iteration {i}")
    
    netEnc.train(), netDec.train(), netD.train()
    G_losses, D_losses = train(netEnc, netDec, netD, train_dataloader, opts, device)
    train_fig = plot_train_hist(D_losses, G_losses)
    train_fig.savefig(opts.model_out / f"img/train_hist_iter{i}")
    
    # Anomaly Detection
    print("\n> Anomaly Detection")
    eval_dataloader = setup_test_dataset(train_dataloader.dataset, anomalies)
    
    netEnc.train(False), netDec.train(False), netD.train(False)
    # netEnc.model = nn.Sequential(*(l for l in netEnc.model if type(l).__name__ != "BatchNorm1d"))
    # netDec.model = nn.Sequential(*(l for l in netEnc.model if type(l).__name__ != "BatchNorm1d"))
    # print(netEnc, netDec)
    anomaly_scores = reconstruction_anomaly(netDec, netEnc, eval_dataloader, device)
    
    # thresh = opts.anomaly_threshold / (i+1) # decrease percentage of data that is dropped in every iter
    thresh = opts.anomaly_threshold if i == 0 else 0.01
    print("Anomaly Percentage: ", thresh)
    score_thresh , _ = plot_anomaly_scores(anomaly_scores, thresh, opts.data_path, opts.model_out /  f"img/iter{i}_reconstr_scores",
                        scale="log", method=f"Reconstruction Iter {i}")
    detected_anomalies = [score[0] for score in anomaly_scores if score[1] > score_thresh]
    anomalies.extend(detected_anomalies)
    logger.log(f"Anomalies detected in Iter {i}: {len(detected_anomalies)}")
    plt.close('all')
    
final = setup_train_dataset(anomalies, opts.iters)
logger.log("Number of anomalies detected: " + str(len(anomalies)))
logger.log("Final size of dataset: " + str(len(final.dataset)))
logger.close()
print("Done.")
    