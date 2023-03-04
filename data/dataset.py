from torch.utils.data import Dataset
import torch
import pandas as pd
import numpy as np
from sklearn.preprocessing import QuantileTransformer, RobustScaler, PowerTransformer, MinMaxScaler
from pathlib import Path

from tools.viz import plot_data_values, plot_single_var

class MULTI_VP_Dataset(Dataset):
    def __init__(self, path : Path, method : str = 'multi', remove_extreme=False, scaler = MinMaxScaler(), nseqs : int = 4, sequences : bool = True) -> None:
        super().__init__()
        
        self.path = path
        self.method = method
        self.scaler = None
        self.stride = 1
        self.nseqs = nseqs
        self.sequences = sequences
        
        # read from inputs compilation
        self.inputs = pd.read_csv(path)
        
        if remove_extreme: self._remove_extreme()
             
        self.filenames = self.inputs['filename'].values
        self.inputs = self.inputs.iloc[:, 1:] # remove filename column
        self.inputs.columns = self.inputs.columns.astype(int) # convert column names to int
        
        if method == "single_mag":
            self.inputs = self.inputs.iloc[:, 640:1280]
        
        # scale data
        self.scaler = scaler
        self.inputs = self.scaler.fit_transform(self.inputs)[:2500]
        
        if self.sequences:
            self.seq_len = len(self.inputs) // nseqs
            self.create_sequences()
        # print(self.inputs[0].shape)
        
            
        # reshape inputs to the desired format
        # self._reshape_inputs(method, nseqs)
        print("Inputs shape:", self.inputs.shape)
        print("Inputs head:\n", self.inputs[:5])
        
    def __len__(self):
        return len(self.inputs)
    
    def __getitem__(self, idx):
        return torch.tensor(self.inputs[idx]).float(), self.filenames[idx]
    
    
    def create_sequences(self):
        input_seqs = []
        for i in range(0, len(self.inputs)-self.seq_len, self.stride):
            input_seqs.append(self.inputs[i:i+self.seq_len])
            # input_seqs.append(self.inputs[i:i+self.seq_len])

        self.inputs = np.array(input_seqs)
            
        
    
    def _reshape_inputs(self, method, nseqs):
        # [
        #     [[R1], [B1], [alpha1]], 
        #     [[R2]. [B2], [alpha2]]
        # ]
        if method == 'multi':
            reshaped = []
            for row in self.inputs:
                reshaped.append([
                    row[:640],
                    row[640:1280],
                    row[1280:1920],
                ])
            self.inputs = np.array(reshaped)
        # [
        #     [R1,B1,alpha1], 
        #     [R2,B2,alpha2]
        # ]
        elif method == 'joint':
            self.inputs = self.inputs.to_numpy()
            self.inputs = self.inputs.reshape(self.inputs.shape[0], 1, self.inputs.shape[1]) # 2d -> 3d
        # [
        #     [[B1_s1], [B1_s2], [B1_sn]], 
        #     [[B2_s1], [B2_s2], [B2_sn]]
        # ]
        elif method == 'single_mag':
            reshaped = []
            # split magntic field into nseqs sequences
            for line in self.inputs:
                reshaped.append(np.split(line, nseqs))
            self.inputs = np.array(reshaped)
            
    def _remove_extreme(self):
        # bad_files = ["profile_wso_CR1992_line_1504"]
        
        initial_len = len(self.inputs)
        # self.inputs = self.inputs[~self.inputs['filename'].isin(bad_files)]
        
        # remove extreme values based on magnetic field
        bad_indices = []
        for i, row in self.inputs.iterrows():
            mags = np.abs(row.iloc[640:1280])
            if (mags.max() > 110 or 
                mags[400:].max() > 1 or
                mags[300:].max() > 30):
                bad_indices.append(i)
            
        
        self.inputs.drop(bad_indices, inplace=True)
        print("Removed {} extreme values".format(initial_len-len(self.inputs)))
    
    def unscale(self, values):
        
        # get first from strided windows
        if self.sequences:
            return self.scaler.inverse_transform([vals[0] for vals in values])
        
        unscaled = []
        if not self.scaler:
            unscaled = values.reshape(values.shape[0], -1)
        
        if self.method == 'multi':
            unscaled = self.scaler.inverse_transform(
                np.array([np.concatenate(vals, axis=0) for vals in values])
            )
        elif self.method == 'joint':
            unscaled = self.scaler.inverse_transform(
                values.reshape(values.shape[0], -1)
            )
        elif self.method == 'single_mag':
            unscaled = self.scaler.inverse_transform(
                values.reshape(values.shape[0], -1)
            )
            
        return unscaled

            
    def plot(self, title : str = "MULTI-VP Data", **figkwargs):
        
        unscaled_inputs = self.unscale(self.inputs)
        print(unscaled_inputs.shape)
        # unscaled_inputs = self.inputs
        if self.method == 'multi':
            plot_data_values(unscaled_inputs, title, scales={'B [G]':'log', 'alpha [deg]': 'linear'}, **figkwargs)
        else:
            plot_single_var(unscaled_inputs, title, scale="log", label="B [G]", **figkwargs)
            
    # def plot_values(self, data : np.ndarray = None, title : str = "MULTI-VP Data", **figkwargs):
    #     unscaled_inputs = self.unscale(data)
    #     if self.method == 'multi':
    #         plot_data_values(unscaled_inputs, title, scales={'B [G]':'log', 'alpha [deg]': 'linear'}, **figkwargs)
    #     else:
    #         plot_single_var(unscaled_inputs, title, scale="log", **figkwargs)
        
    