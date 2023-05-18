import numpy as np
from argparse import ArgumentParser
from pathlib import Path
import pandas as pd

if __name__ == "__main__":
    
    parser = ArgumentParser()
    parser.add_argument("--data-root", type=Path, default="./data/MULTI_VP_Profiles")
    parser.add_argument("--proc-dir", type=Path, default="./data/processed")
    parser.add_argument("--abs", action="store_true")
    parser.add_argument("--val-files", type=Path)
    
    opts = vars(parser.parse_args())
    if not opts['proc_dir'].exists():
        opts['proc_dir'].mkdir(parents=True)
    
    val_files = list()
    if opts["val_files"]:
        with open(opts["val_files"], "r") as f:
            val_files = f.readlines()
            val_files = [f.split(".")[0] for f in val_files]
        val_files = set(val_files)

    data_dirs = [d for d in opts['data_root'].iterdir() if d.is_dir()]
    n_bad = 0
    
    files = []
    for d in data_dirs:
        files.extend([f for f in d.iterdir() if f.is_file()])
    
    
    
    if len(val_files) > 0:
        files = [f for f in files if f.stem in val_files]
    
    for f in files:
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
            
        if len(val_files) == 0 and flag: 
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
                out.writelines(lines[2:])
        
        # apply abs on magnetic field
        if opts['abs']:
            df = pd.read_csv(f"{opts['proc_dir'] / f.stem}.csv", sep=',')
            df["B [G]"] = df["B [G]"].abs()
            df.to_csv(f"{opts['proc_dir'] / f.stem}.csv", index=False)
                    
    print("\nFinal number of files: ", len([f for f in opts['proc_dir'].iterdir()]))
    print("Number of bad files: ", n_bad)