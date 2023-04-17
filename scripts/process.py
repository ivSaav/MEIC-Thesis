import numpy as np
from argparse import ArgumentParser
from pathlib import Path
import pandas as pd

if __name__ == "__main__":
    
    parser = ArgumentParser()
    parser.add_argument("--data-root", type=Path, default="./data/MULTI_VP_Profiles")
    parser.add_argument("--proc-dir", type=Path, default="./data/processed")
    parser.add_argument("--abs", action="store_true")
    
    opts = vars(parser.parse_args())
    if not opts['proc_dir'].exists():
        opts['proc_dir'].mkdir(parents=True)
        
    data_dirs = [d for d in opts['data_root'].iterdir() if d.is_dir()]
    n_bad = 0
    for d in data_dirs:
        for f in d.iterdir():
            print("File : ", f, end="\r", flush=True)
            df = pd.read_csv(f, skiprows=2, sep=',')
            
            flag = 0
            if not(df.isnull().values.any()):
                for j in df.columns:
                    if(df[j].dtype != np.int64 and df[j].dtype != np.float64 ): #verifying if there's no weird data. if there is, skip
                        flag = 1
                        break
            else: 
                flag = 1
                
            if flag: 
                n_bad += 1
                print("File: ", f, " is corrupted.")
                continue
            
            # remove commented header
            with open(f, 'r') as tmp:
                lines = tmp.readlines()
                values = lines[2].replace('#', '').split(',')
                values = [v.strip() for v in values]
                lines[2] = ','.join(values) + '\n'

                with open(f"{opts['proc_dir'] / f.stem}.csv" , 'w') as out:
                    out.writelines(lines)
            
            # apply abs on magnetic field
            if opts['abs']:
                df = pd.read_csv(f"{opts['proc_dir'] / f.stem}.csv", skiprows=2, sep=',')
                df["B [G]"] = df["B [G]"].abs()
                df.to_csv(f"{opts['proc_dir'] / f.stem}.csv", index=False)
                    
    print("\nFinal number of files: ", len([f for f in opts['proc_dir'].iterdir()]))
    print("Number of bad files: ", n_bad)