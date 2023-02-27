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
        # read from inputs compilation
        self.inputs = pd.read_csv(path)
        
        if remove_extreme:
            self._remove_extreme(self.inputs)
             
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
    
    def _remove_extreme(self, inputs, labels):
        # remove extreme values based on magnetic field
        self.inputs.drop(self.inputs.loc[self.inputs.iloc[:, 640:1280] > 110].index, inplace=True)
        # TODO
        pass
    
    def _plot(self, title : str = "MULTI-VP Data", **figkwargs):
        tmp_inputs = self.scaler.inverse_transform(self.inputs)
        plot_data_values(tmp_inputs, title, scales={'B [G]':'symlog', 'alpha [deg]': 'linear'}, **figkwargs)
        
    