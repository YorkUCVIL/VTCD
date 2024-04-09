#!/bin/bash
# BVH, Jun 2022.

# conda activate bcv11

for i in {1..11}
do

echo $i

python gen_dset/cv_export_kubcon_v4.py

rm -rf /tmp/*

done

# pkill -9 python
