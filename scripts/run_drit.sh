#! /bin/bash

cd /sinergia/benchmark
./train.py --dataset-type MultiDomain --model DRIT -e 100 --save -v --evaluate --evaluation-frequency 1 --save-frequency 1 --name DRIT_gta_cityscapes --input-size 216  --domain-folders gta5/images,cityscapes -d gta,cityscapes --data-root .. --num-threads 4 --max-dataset-size 10000
