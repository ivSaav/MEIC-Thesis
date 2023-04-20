import os
from pathlib import Path
from pickle import load


clusters = [f for f in Path("local").iterdir() if f.is_file()]

for conf in clusters:
    with open(conf, 'rb') as cf:
        all_runs = load(cf)

    for run_dict in all_runs:
        run_id = run_dict['run_id']

        print("Executing: ", f"python train_eval.py -d ../data/compiled -m models -cf {conf} -r {run_id}")
        os.system(f"python train_eval.py -d ../data/compiled -m models -cf {conf} -r {run_id}") 
