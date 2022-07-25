#!/bin/bash
cd $ROBOT
cd fridge_dataset
echo $(pwd)
echo "Removing old dataset /data/dataset/pascal/pascalparts_person"
rm -rf /data/dataset/pascal/pascalparts_person
python tools/dataset.py --cfg configs/pascalparts/person_train.yaml &&
python tools/dataset.py --cfg configs/pascalparts/person_val.yaml &&

echo "Run notebook subset_pascalparts_person.ipynb! to get val_subset"
echo "DONE"
