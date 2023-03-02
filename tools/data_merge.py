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
  
  tmp_out = df.iloc[:,[4,5,6]]
  tmp_in = df.iloc[:,[0,7,9]]
  
  ns = tmp_out[['n [cm^-3]']].values
  vs = tmp_out[['v [km/s]']].values
  ts = tmp_out[['T [MK]']].values
  
  file_outputs = np.concatenate((ns,vs,ts), axis=0)
  
  rs = tmp_in[['R [Rsun]']].values
  bs = tmp_in[['B [G]']].values
  alphas = tmp_in[['alpha [deg]']].values
  
  file_inputs = np.concatenate((rs,bs,alphas), axis=0)
  
  inputs.append(pd.DataFrame(file_inputs.T))
  outputs.append(pd.DataFrame(file_outputs.T))
  
  print("[{}/{}] {}".format(idx, nfiles, filename.stem), end="\r", flush=True)
  
  
filename_series = [pd.DataFrame([f.stem for f in files], columns=['filename'])]
data_inputs = pd.concat(inputs, axis=0, ignore_index=True)
data_inputs = pd.concat(filename_series + [data_inputs], axis=1)

data_outputs = pd.concat(outputs, axis=0, ignore_index=True)
data_outputs = pd.concat(filename_series + [data_outputs], axis=1)
  
  
print(data_inputs)
print(data_outputs)
data_inputs.to_csv(output_path / 'inputs.csv', index=False) 
data_outputs.to_csv(output_path / 'outputs.csv', index=False)
