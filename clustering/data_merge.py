import pandas as pd
import numpy as np

from pathlib import Path

output_path = Path("../data/original/compiled/")
if not output_path.exists():
  output_path.mkdir(parents=True)

#files = glob.glob("./MULTI_VP_profiles/profiles_wso_CR1992"+"\\*.csv") 
files = sorted([f for f in Path("../data/original/processed/").iterdir()])
broken_files = set()
# random.shuffle(files)
# n_files = (len(files))
# n_files_validation = int(math.ceil(n_files*0.1))


data_inputs = pd.DataFrame()
data_ns = pd.DataFrame().astype(float)
data_vs = pd.DataFrame().astype(float)
data_ts = pd.DataFrame().astype(float)

for idx, filename in enumerate(files):
  
  df = pd.read_csv(filename,skiprows=2)
  df = df.astype(float)
  
  print("File: ", idx, filename)
  bad_data = 0
  if not(df.isnull().values.any()):
    for j in df.columns:
        if(df[j].dtype != np.int64 and df[j].dtype != np.float64 ): #verifying if there's no weird data. if there is, skip
          bad_data = 1
          broken_files.add(filename)
          break
  else: 
    bad_data = 1
    broken_files.add(filename)
  
  if (bad_data): 
    continue
  
  # select input and output columns
  n = df.iloc[:,4].values.flatten()
  v = df.iloc[:,5].values.flatten()
  t = df.iloc[:,6].values.flatten()
  inputs = df.iloc[:,[0,7,9]].values.flatten() # get all inputs as an array
  
  df_n = pd.Series(n).astype(float)
  df_v = pd.Series(v).astype(float)
  df_t = pd.Series(t).astype(float)
  df_inputs = pd.Series(inputs).astype(float) # 
  
  # print(df_inputs)
  # exit(0)

  data_ns = pd.concat([data_ns, df_n], ignore_index = True, axis = 1)
  data_vs = pd.concat([data_vs, df_v], ignore_index = True, axis = 1)
  data_ts = pd.concat([data_ts, df_t], ignore_index = True, axis = 1)
  data_inputs = pd.concat([data_inputs, df_inputs], ignore_index = True, axis = 1)
  

# Transpose dataframes
data_inputs.columns = [f.stem for f in files if f not in broken_files]
data_inputs = data_inputs.T

print(data_inputs.head())
data_ns = data_ns.T
data_vs = data_vs.T
data_ts = data_ts.T

print("N training\n ", data_ns)

data_ns.to_csv(output_path / 'output_n_data_compilation.csv', index=False) 
data_vs.to_csv(output_path / 'output_v_data_compilation.csv', index=False) 
data_ts.to_csv(output_path / 'output_t_data_compilation.csv', index=False) 
data_inputs.to_csv(output_path / 'inputsdata_compilation.csv', index=True) 

#__________________________________________________________________
#juntar os outputs de treino num ficheiro n1,v1,t1, n2,v2,t2, ...
df_n = pd.read_csv(output_path / 'output_n_data_compilation.csv')
df_v = pd.read_csv(output_path / 'output_v_data_compilation.csv')
df_t = pd.read_csv(output_path / 'output_t_data_compilation.csv') 
df_outputs = pd.DataFrame()  

print(len(df_n),df_n.shape[1])

#print(df_n.head())
for col in range(0,640): 
    column = str(col)
    n_col = df_n[column]
    v_col = df_v[column]
    t_col = df_t[column]
    if (col == 0): 
        df_outputs = pd.concat([n_col,v_col], axis = 1)
        df_outputs = pd.concat([df_outputs, t_col], axis = 1)
    else: 
        df_outputs = pd.concat([df_outputs, n_col, v_col, t_col], axis = 1)
    
print(df_outputs)
print(len(df_outputs),df_outputs.shape[1])
df_outputs.to_csv(output_path / 'outputs_compiled.csv', index=False) 
