from torch.utils.data import Dataset
import torch
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA

from pathlib import Path
from typing import List

from tools.viz import plot_data_values, plot_single_var

class MULTI_VP_Dataset(Dataset):
    def __init__(self, path : Path, method : str = 'multi', remove_extreme=False, is_train : bool = False, 
                 scaler = None, nseqs : int = 0, window_size : int = 0, drop_radius : bool = True,
                 use_pca : bool = False, n_components : int = 32, pca = None, pca_scaler = None) -> None:
        super().__init__()
        
        self.path = path
        self.method = method
        self.scaler = scaler
        self.nseqs = nseqs
        self.wsize = window_size
        self.n_components = n_components
        self.pca = pca
        self.pca_scaler = pca_scaler
        self.drop_radius = drop_radius
        
        # read from inputs compilation
        self.inputs = pd.read_csv(path)
        
        if remove_extreme: self._remove_extreme()
             
        self.filenames = self.inputs['filename'].values
        self.inputs = self.inputs.iloc[:, 1:] # remove filename column
        self.inputs.columns = self.inputs.columns.astype(int) # convert column names to int
        self.length = len(self.inputs)
        
        self._preprocess(is_train, use_pca, drop_radius)
        self.shape = self.inputs.shape
    
    @classmethod
    def _from(cls, old, is_train=False, remove_extreme=False):
        return cls(
            path=old.path,
            method=old.method,
            remove_extreme=remove_extreme,
            is_train=is_train,
            scaler=old.scaler,
            window_size=old.wsize,
            nseqs=old.nseqs,
            use_pca=old.pca != None,
            n_components=old.n_components,
            pca=old.pca,
            pca_scaler=old.pca_scaler,
            drop_radius=old.drop_radius
        )        

    def __len__(self):
        return self.length
    
    def __getitem__(self, idx):
        if self.method == "window_mag":
            window = self.inputs[idx:idx+self.wsize]
            # return window and name of the starting filename
            return torch.tensor(window).float(), self.filenames[idx]
        return torch.tensor(self.inputs[idx]).float(), self.filenames[idx]
    
    
    def _preprocess(self, is_train : bool, use_pca : bool, drop_radius : bool):
        # select only magnetic field values
        if self.method == "single_mag" or self.method == "window_mag":
            self.inputs = self.inputs.iloc[:, 640:1280]
            self.inputs.columns = range(self.inputs.shape[1])
        elif self.method == "single_alpha":
            self.inputs = self.inputs.iloc[:, 1280:]
            self.inputs.columns = range(self.inputs.shape[1])
        elif drop_radius:
            self.inputs = self.inputs.iloc[:, 640:]
        
        # only fit data when it is normal
        if is_train:
            self.inputs = self.scaler.fit_transform(self.inputs)
            if use_pca: 
                self.pca = PCA(n_components=self.n_components).fit(self.inputs)
                self.inputs = self.pca.transform(self.inputs)
                # self.pca_scaler = StandardScaler()
                self.inputs = self.pca_scaler.fit_transform(self.inputs)
        else: # test
            self.inputs = self.scaler.transform(self.inputs)
            if use_pca: 
                self.inputs = self.pca.transform(self.inputs)
                self.inputs = self.pca_scaler.transform(self.inputs)
                
        if self.method == "window_mag":
            self.length = self.length - self.wsize # update length to accomodate window size
            first, _ = self.__getitem__(0) # sanity check
            print(f"Window size: {self.wsize}")
            print("Window shape: ", first.shape)
            print("First window:\n", first)
        else:
            self._reshape_inputs(self.method, self.nseqs)
            
        print("Inputs shape:", self.inputs.shape)
        print("Inputs head:\n", self.inputs[:5])
        
         
    
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
        # elif method == 'joint':
        #     self.inputs = self.inputs.reshape(self.inputs.shape[0], self.inputs.shape[1]) # 2d -> 3d
        # [
        #     [[B1_s1], [B1_s2], [B1_sn]], 
        #     [[B2_s1], [B2_s2], [B2_sn]]
        # ]
        # elif method == 'single_mag':
        #     reshaped = []
        #     # split magntic field into nseqs sequences
        #     for line in self.inputs:
        #         reshaped.append(np.split(line, nseqs))
        #     self.inputs = np.array(reshaped)
     
            
    def _remove_extreme(self) -> None:
        initial_len = len(self.inputs)
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
    
    
    def _get_filename_indexes(self, files : List[str]):
        files = set(files)
        indexes = []
        for idx, f in enumerate(self.filenames):
            if f in files:
               indexes.append(idx)    
        return indexes

    def remove_files(self, files : List[str]) -> None:
        """Remove files from the dataset

        Args:
            files (list): list of files to remove
        """ 
        indexes = self._get_filename_indexes(files)  
        original_size = self.inputs.shape[0]
        # indexes = np.in1d(self.filenames, np.array(files)).nonzero()[0]
        print("Number of files to remove: ", len(files))
        self.inputs = np.delete(self.inputs, indexes, axis=0)
        self.length = len(self.inputs)
        self.filenames = np.delete(self.filenames, indexes, axis=0)
        print("Removed ", original_size-self.inputs.shape[0], " files")
        
        if (self.length != len(self.filenames)):
            print("Error: length of inputs and filenames do not match")
            exit(1)
        # print("Removed {} files".format(len(files)))
    
    def filter_profiles(self, profiles : List[str]) -> None:
        if not profiles: return []
        def filter_profile(f : str) -> bool:
            for p in profiles:
                if p in f: return False
            return True
        files = list(filter(filter_profile, self.filenames))
                    
        self.remove_files(files)
        return files
           
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

            
    def plot(self, title : str = "MULTI-VP Data", scaled : bool = False, **figkwargs) -> None:
        """Plot input values

        Args:
            title (str, optional): title of the plot. Defaults to "MULTI-VP Data".
        """
        
        if scaled:
            unscaled_inputs = self.flatten(self.inputs)
        else:
            unscaled_inputs = self.unscale(self.inputs)
            
        print("Unscaled inputs shape:", self.inputs.shape)
        if self.method == 'multi' or self.method == 'joint':
            labels = ["R [Rsun]", "B [G]", "alpha [deg]"] if not self.drop_radius else ["B [G]", "alpha [deg]"]
            plot_data_values(unscaled_inputs, title, labels=labels, 
                            scales={"B [G]" : "log", "alpha [deg]" : "symlog"}, **figkwargs)
            
        else:
            plot_single_var(unscaled_inputs, title, scale="log", label="B [G]", **figkwargs)