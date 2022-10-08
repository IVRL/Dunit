#! /bin/bash

cd /sinergia/benckmark
./train.py --dataset-type SourceTarget --model UNITObjectDetection -e 100 -v --save --evaluate --evaluation-frequency 1 --save-frequency 1 --name UNIT_object_detection_init --max-dataset-size 10000 --source-annotation-type init --target-annotation-type init --source-annotation init/ --target-annotation init/ --with-annotations --pool-size=50 --input-size 256 --normalize --data-root .. --num-threads=4 --source init --target init

