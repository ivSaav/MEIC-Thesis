#!/bin/bash

for file in ./local/*
do
    if [[ -f $file ]]; then
        python train_eval.py -d ../data/compiled/ -m out -cf $file
    fi
done
