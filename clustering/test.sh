#!/bin/bash

for  file in ./clusters/*
do
    if [[ -f $file ]]; then
        python train_eval.py -d ../data/compiled/ -m models -cf $file
    fi
done
