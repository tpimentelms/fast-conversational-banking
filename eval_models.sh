#!/bin/bash
if [ $# -lt 2 ] 
    then
    echo "Not enough arguments supplied. Should be used as:"
    echo "    $ source eval_models.sh <folder_with_models> <path_to_data>"
    return
fi

eval_time=''
if [ $# -gt 2 ] 
    then
    eval_time='--eval-time'
fi

for d in $1*/ ; do
    echo "Model: "$d $eval_time
    python -u eval.py --data $2 --batch-size 16 --load-model $d  $eval_time
done
