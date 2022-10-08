#! /bin/bash

cd /sinergia/benchmark
./train.py --dataset-type SourceTarget --model DRITObjectDetection -e 2 --nb-pretraining-epochs 1 --save --evaluate --evaluation-frequency 1 --save-frequency 1 --name test_INIT --max-dataset-size 4 --source-annotation-type init --target-annotation-type init --source-annotation source/fake_init.txt --target-annotation target/fake_init.txt --with-annotations --pool-size=2 --input-size 216 --normalize --batch-size 2 --verbose
