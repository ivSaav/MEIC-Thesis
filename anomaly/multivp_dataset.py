from torch.utils.data import Dataset
import torch
import pandas as pd
import numpy as np
from sklearn.preprocessing import QuantileTransformer, RobustScaler, PowerTransformer, MinMaxScaler
from pathlib import Path

from tools.viz import plot_data_values, plot_single_var

class MULTI_VP_Dataset(Dataset):
    def __init__(self, path : Path, method : str = 'multi', remove_extreme=False, scaler = MinMaxScaler(), 
                 nseqs : int = 4, window_size : int = 1, pca : bool = False) -> None:
        super().__init__()
        
        self.path = path
        self.method = method
        self.scaler = None
        self.nseqs = nseqs
        self.scaler = scaler
        
        # read from inputs compilation
        self.inputs = pd.read_csv(path)
        
        if remove_extreme: self._remove_extreme()
             
        self.filenames = self.inputs['filename'].values
        self.inputs = self.inputs.iloc[:, 1:] # remove filename column
        self.inputs.columns = self.inputs.columns.astype(int) # convert column names to int
        self.length = len(self.inputs)
        
        if method == "single_mag" or method == "window_mag":
            self.inputs = self.inputs.iloc[:, 640:1280]
            self.inputs.columns = range(self.inputs.shape[1])
        
        # transform inputs
        self.inputs = self.scaler.fit_transform(self.inputs)
        if self.method == "window_mag":
            self.length = self.length - window_size # update length to accomodate window size
            self.wsize = window_size
            first, _ = self.__getitem__(0) # sanity check
            print(f"Window size: {window_size}")
            print("Window shape: ", first.shape)
            print("First window:\n", first)
        elif self.method == "multi":
            self._reshape_inputs(method, nseqs)
        print("Inputs shape:", self.inputs.shape)
        print("Inputs head:\n", self.inputs[:5])
        
    def __len__(self):
        return self.length
    
    def __getitem__(self, idx):
        if self.method == "window_mag":
            window = self.inputs[idx:idx+self.wsize]
            # return window and name of the starting filename
            return torch.tensor(window).float(), self.filenames[idx]
        
        return torch.tensor(self.inputs[idx]).float(), self.filenames[idx]
         
    
    def _reshape_inputs(self, method : str, nseqs : int) -> None:
        """Reshape inputs to the desired shape

        Args:
            method (str): input method
            nseqs (int): number of sequences to split the input into
        """
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
     
            
    def _remove_extreme(self) -> None:
        initial_len = len(self.inputs)
        # bad_files = ["profile_wso_CR1992_line_1504"]
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
    
    
    def unscale(self, values : np.ndarray) -> np.ndarray:
        """Inverse transform values

        Args:
            values (np.ndarray): values to be unscaled

        Returns:
            np.ndarray: unscaled values
        """
        return self.scaler.inverse_transform(self.flatten(values))
    
    
    def flatten(self, values : np.ndarray) -> np.ndarray:
        """Flatten values to the original input shape

        Args:
            values (np.ndarray): values to reshape

        Returns:
            np.ndarray: reshaped values
        """
        # get first element of every window
        if self.method == "window_mag":
            return np.array([vals[0] for vals in values])
        return values.reshape(values.shape[0], -1)
    #TODO might be different for "multi"

            
    def plot(self, title : str = "MULTI-VP Data", **figkwargs) -> None:
        """Plot input values

        Args:
            title (str, optional): title of the plot. Defaults to "MULTI-VP Data".
        """
        unscaled_inputs = self.inputs
        # unscaled_inputs = self.unscale(self.inputs)
        if self.method == 'multi' or self.method == 'joint':
            plot_data_values(unscaled_inputs, title, scales={'B [G]':'log', 'alpha [deg]': 'linear'}, **figkwargs)
        else:
            plot_single_var(unscaled_inputs, title, scale="linear", label="B [G]", **figkwargs)
        
    