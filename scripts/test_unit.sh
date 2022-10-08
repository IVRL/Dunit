#! /bin/bash

cd /sinergia/benchmark
./test.py --model UNIT -v --name UNIT_gta_cityscapes --batch-size 10 --source gta5/images/test --target cityscapes/test --data-root ..
