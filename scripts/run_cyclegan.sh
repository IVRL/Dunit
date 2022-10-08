#! /bin/bash

cd /sinergia/benchmark
./train.py --dataset-type SourceTarget --model CycleGAN -e 100 --save -v --evaluate --evaluation-frequency 1 --save-frequency 1 --name CycleGAN_gta_cityscapes --batch-size 1 --source gta5/images --target cityscapes --data-root ..
