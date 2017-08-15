#!/bin/bash 
dataset='samantha'
min_count=1
remove_brackets=''
python -m data_handler.create_dataset $remove_brackets \
    --train-src "data/"$dataset"/pt-log.pt.txt" --train-tgt "data/"$dataset"/pt-log.log.txt" \
    --min-count $min_count --output-file "data/"$dataset"/dataset"$remove_brackets".pckl"
