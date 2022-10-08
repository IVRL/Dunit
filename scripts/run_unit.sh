#! /bin/bash

cd /sinergia/benchmark
./train.py --dataset-type SourceTarget --model UNIT -e 100 --save -v --evaluate --evaluation-frequency 1 --save-frequency 1 --name UNIT_gta_cityscapes --batch-size 10 --source gta5/images --target cityscapes --data-root ..
