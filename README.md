# Conversational Banking


## Setup

With CUDA 8.0:
```bash
$ conda create --name conv-bank python=3.6.1
$ source activate conv-bank
$ conda install pytorch torchvision cuda80 -c soumith
<!-- $ pip install ipdb -->
```

Without CUDA:
```bash
$ conda create --name conv-bank python=3.6.1
$ source activate conv-bank
$ conda install pytorch torchvision -c soumith
<!-- $ pip install ipdb -->
```

### Model options

* cnn
* full-cnn
* reduce-cnn
* full-reduce-cnn
* single-reduce-cnn
* double-reduce-cnn
* bytenet
* rnn
* gru
* attn-gru

## Train model


```bash
$ python -u train.py --src data/samantha/pt-log.pt.txt --tgt data/samantha/pt-log.log.txt --epochs 1 --model cnn --dilate 2
```


## Eval model

```bash
$ source eval_models.sh <path-to-models> <path-to-data>
```

```bash
$ source eval_models.sh checkpoint/samantha/double-cnn-reduce-1000__layers_1__kernel_size_16__stride_1__ignore_pad data/samantha/
```


## ETC


```bash
model='full-cnn'
dataset='jobqueries'
epochs=5000000
save_every=20
single_linear=''
batch_size=16
weight_decay=0.0001
j=3
k=0.5
i=512
min_count=2
file_name=$dataset"/"$model"-epochs-"$epochs"__kernelsize_"$j"__hiddensize_"$i"__dilate_2__dropout_"$k"__weight-decay_"$weight_decay"__save-every_"$save_every"__batch-size_"$batch_size"__min-count_"$min_count"__ignore_pad"$single_linear
echo $dataset $model $epochs -hidden-size $i -kernel-size $j --dropout $k --weight-decay $weight_decay --min-count $min_count --save-every $save_every --ignore-pad $single_linear
python -u train.py --data data/$dataset/dataset_min-$min_count.pckl \
                    --epochs $epochs --model $model --batch-size $batch_size \
                    --hidden-size $i --kernel-size $j --dilate 2 --dropout $k --weight-decay $weight_decay \
                    --save-every $save_every $single_linear --ignore-pad \
                    --save-dir "checkpoint/"$file_name"/" > "log/"$file_name".txt"
<!-- python -u train.py --src data/$dataset/train.en.txt --tgt data/$dataset/train.log.txt \
                    --epochs $epochs --model $model --batch-size $batch_size \
                    --hidden-size $i --kernel-size $j --dilate 2 --dropout $k --weight-decay $weight_decay \
                    --save-every $save_every $single_linear --ignore-pad \
                    --save-dir "checkpoint/"$file_name"/" > "log/"$file_name".txt" -->
<!-- python -u eval.py --src data/$dataset/train.en.txt --tgt data/$dataset/train.log.txt --batch-size $batch_size --load-model "checkpoint/"$file_name"/" -->
python -u eval.py --data data/$dataset/dataset_min-$min_count.pckl --batch-size $batch_size --load-model "checkpoint/"$file_name"/"
```
