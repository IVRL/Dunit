#! /bin/bash
OPTIONS="--save -v --evaluate --evaluation-frequency 1 --save-frequency 1 --batch-size 2 --max-dataset-size 4"

# UNIT
UNIT="--dataset-type SourceTarget --model UNIT -e 2 --name test_UNIT $OPTIONS"

# MUNIT
MUNIT_ST="--dataset-type SourceTarget --model MUNIT -e 2 --name test_MUNIT_multi_domain $OPTIONS"
MUNIT_MULTI="--dataset-type MultiDomain --model MUNIT -e 2 --name test_MUNIT_multi_domain $OPTIONS"

# CycleGAN
CYCLEGAN="--dataset-type SourceTarget --model CycleGAN -e 2 --name test_CycleGAN $OPTIONS"

# DRIT on small dataset
DRIT="-e 2 --dataset-type MultiDomain --model DRIT --name test_DRIT --input-size 216 $OPTIONS"
# test resuming
RESUME="--dataset-type MultiDomain --model DRIT -e 4 --name test_DRIT --input-size 216 --resume $OPTIONS"

# UNIT with object detection
UNIT_OBJ="-e 2 --nb-pretraining-epochs 2 --dataset-type SourceTarget --model UNITObjectDetection --name test_INIT --source-annotation-type init --target-annotation-type init --source-annotation source/fake_init.txt --target-annotation target/fake_init.txt --with-annotations --pool-size=2 --input-size 256 --normalize $OPTIONS"

./train.py $UNIT && ./train.py $MUNIT_ST && ./train.py $MUNIT_MULTI && ./train.py $CYCLEGAN && ./train.py $DRIT && ./train.py $RESUME && ./train.py $UNIT_OBJ
