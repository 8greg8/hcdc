#!/bin/bash
cd $ROBOT
cd fridge_dataset

path=/data/dataset/grmix/orig/annotations &&
python tools/tooccclassification.py --coco $path/test_wholes.json --occ occupancy &&

python tools/dataset.py --cfg configs/aug/grmix_train.yaml &&

echo "DONE"