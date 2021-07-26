#!/bin/bash

# conda activate pytorch

mkdir results
mkdir data
python dataset.py

for name in {heart_rotate,heart_scale,heart_embed}
do
   mkdir results/heart_$name
   python create_log_file.py heart $name
   for i in {1..10}
   do
    python nn.py $i heart $name
   done
done

