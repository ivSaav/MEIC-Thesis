from torch.utils.data import Dataset
import torch
import pandas as pd
import numpy as np
from sklearn.preprocessing import QuantileTransformer
from pathlib import Path

from tools.viz import plot_data_values

class MULTI_VP_Dataset(Dataset):
    def __init__(self, path : Path, method : str = 'multi', remove_extreme=False) -> None:
        super().__init__()
        
        self.path = path
        self.method = method
        
        # read from inputs compilation
        self.inputs = pd.read_csv(path)
        
        if remove_extreme: self._remove_extreme()
             
        self.filenames = self.inputs['filename'].values
        self.inputs = self.inputs.iloc[:, 1:] # remove filename column
        self.inputs.columns = self.inputs.columns.astype(int) # convert column names to int
        
        # scale data
        self.scaler = QuantileTransformer()
        self.inputs = self.scaler.fit_transform(self.inputs)
        
        if method == 'multi':
            reshaped = []
            for row in self.inputs:
                reshaped.append([
                    row[:640],
                    row[640:1280],
                    row[1280:1920],
                ])
            
            self.inputs = torch.from_numpy(np.array(reshaped)).float()
        elif method == 'single':
            self.inputs = torch.from_numpy(self.inputs).float()
            self.inputs = self.inputs.reshape(self.inputs.shape[0], 1, self.inputs.shape[1]) # 2d -> 3d
        
        print("Inputs shape:", self.inputs.shape)
        print("Inputs head:\n", self.inputs[:5])
        
    def __len__(self):
        return len(self.inputs)
    
    def __getitem__(self, idx):
        return self.inputs[idx], self.filenames[idx]
    
    def _remove_extreme(self):
        # remove extreme values based on magnetic field
        bad_indices = []
        for i, row in self.inputs.iterrows():
            mags = np.abs(row.iloc[640:1280])
            if mags.max() > 110:
                bad_indices.append(i)
            if mags[400:].max() > 1:
                bad_indices.append(i)
        
        self.inputs.drop(bad_indices, inplace=True)
        print("Removed {} extreme values".format(len(bad_indices)))
    
    def _unscale(self):
        if self.method == 'multi':
            return self.scaler.inverse_transform(
                np.array([np.concatenate(values, axis=0) for values in self.inputs.numpy()])
            )
        # TODO other methods
        
    
    def plot(self, title : str = "MULTI-VP Data", **figkwargs):
        unscaled_inputs = self._unscale()
        plot_data_values(unscaled_inputs, title, scales={'B [G]':'symlog', 'alpha [deg]': 'linear'}, **figkwargs)
        
    