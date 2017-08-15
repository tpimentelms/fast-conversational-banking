#!/bin/bash 
if [ $# -lt 1 ] 
    then
    echo "Not enough arguments supplied. Should be used as:"
    echo "    $ source train_all.sh <epochs>"
    return
fi

epochs=$1

for model in fasp bytenet redfasp redmulnet-1 redmulnet-2 attn-rnn rnn
do
    echo "Model: "$model
    python train.py --model $model --epochs $epochs --save-dir checkpoint/samantha/$model > "log/samantha/"$model".txt"
done

for model in fasp redfasp bytenet
do
    echo "Model: "$model" (ml)"
    python train.py --model $model --multi-linear --epochs $epochs --save-dir "checkpoint/samantha/"$model"-ml" > "log/samantha/"$model"-ml.txt"
done