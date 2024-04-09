#!/bin/bash
# BVH, Juk 2022.

# conda activate bcv11

for i in {1..41}
do

echo $i

python gen_dset/export_kubcon_v7.py

rm -rf /tmp/*

done

# pkill -9 python
