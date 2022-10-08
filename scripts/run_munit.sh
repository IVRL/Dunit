#! /bin/bash

cd /sinergia/benchmark
./train.py --dataset-type SourceTarget --model MUNIT -e 100 --save -v --evaluate --evaluation-frequency 1 --save-frequency 1 --name MUNIT_gta_cityscapes --source gta5/images --target cityscapes --data-root .. --max-dataset-size 10000
