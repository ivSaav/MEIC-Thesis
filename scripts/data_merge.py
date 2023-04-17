import pandas as pd
import numpy as np

from pathlib import Path

output_path = Path("./data/compiled/")
if not output_path.exists():
  output_path.mkdir(parents=True)

files = sorted([f for f in Path("./data/processed/").iterdir()])

data_inputs = pd.DataFrame()
data_outputs = pd.DataFrame()

inputs, outputs = [], []
nfiles = len(files)
for idx, filename in enumerate(files):
  df = pd.read_csv(filename)
  df = df.astype(float)
  
  ns = df[['n [cm^-3]']].values
  vs = df[['v [km/s]']].values
  ts = df[['T [MK]']].values
  
  # file_outputs = np.concatenate((ns,vs,ts), axis=0)
  file_outputs = []
  for n,v,t in zip(ns,vs,ts):
    file_outputs.extend([n,v,t])
  
  
  rs = df[['R [Rsun]']].values
  bs = df[['B [G]']].values
  alphas = df[['alpha [deg]']].values
  
  file_inputs = np.concatenate((rs,bs,alphas), axis=0)
  
  inputs.append(pd.DataFrame(file_inputs.T))
  outputs.append(pd.DataFrame(file_outputs).T)
  
  print("[{}/{}] {}".format(idx, nfiles, filename.stem), end="\r", flush=True)
  
  
filename_series = [pd.DataFrame([f.stem for f in files], columns=['filename'])]
data_inputs = pd.concat(inputs, axis=0, ignore_index=True)
data_inputs = pd.concat(filename_series + [data_inputs], axis=1)

data_outputs = pd.concat(outputs, axis=0, ignore_index=True)
data_outputs = pd.concat(filename_series + [data_outputs], axis=1)
  
  
print(data_inputs)
print(data_outputs)
data_inputs.to_csv(output_path / 'inputs.csv', index=False) 
data_outputs.to_csv(output_path / 'outputs_inter.csv', index=False)
