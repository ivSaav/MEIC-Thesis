from pickle import dump
from sklearn.preprocessing import QuantileTransformer
import pandas as pd
from typing import Tuple, List
from pathlib import Path

import matplotlib.pyplot as plt

def load_original_data(data_path: Path, save_scalers : bool = False) -> Tuple[pd.DataFrame, pd.DataFrame, QuantileTransformer, QuantileTransformer]:
    """Load the original data from the file."""

    inputs = pd.read_csv(data_path / 'inputsdata_compilation.csv')
    outputs = pd.read_csv(data_path / 'outputsdata_compilation.csv')
    
    input_filenames = inputs[['filename']]
    output_filenames = outputs[['filename']]
    
    scaler_inputs, scaler_ouputs = QuantileTransformer(), QuantileTransformer()
    inputs = scaler_inputs.fit_transform(inputs.iloc[:, 1:])
    outputs = scaler_ouputs.fit_transform(outputs.iloc[:, 1:])
    
    if save_scalers:
        dump((scaler_inputs, scaler_ouputs), open(data_path / 'scalers.pkl', 'wb'))
    
    inputs = pd.DataFrame(inputs)
    inputs = pd.concat([input_filenames, inputs], axis=1)
    
    outputs = pd.DataFrame(outputs)
    outputs = pd.concat([output_filenames, outputs], axis=1)
    
    print("Scaled inputs:", inputs.head())
    print("Scaled outputs:", outputs.head())
    return inputs, outputs, scaler_inputs, scaler_ouputs


def join_files_in_cluster(cluster_files: List[Path], input_data : pd.DataFrame, output_data : pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Join all files in a cluster into a single dataframe."""
    cluster_inputs, cluster_outputs = pd.DataFrame(), pd.DataFrame()
    
    inputs = [input_data.loc[input_data['filename'] == f].iloc[:, 1:]
              for f in cluster_files]
    
    cluster_inputs = pd.concat(inputs, axis=0, ignore_index=True)
    
    outputs = [output_data.loc[output_data['filename'] == f].iloc[:, 1:]
               for f in cluster_files]
    cluster_outputs = pd.concat(outputs, axis=0, ignore_index=True)      
    
    print(cluster_inputs.head())
    print(cluster_inputs.shape)
    # print(cluster_df)
    # print(cluster_df.shape)
    # print(cluster_df.columns)
    print("Cluster shape:", cluster_inputs.shape)
    return cluster_inputs, cluster_outputs