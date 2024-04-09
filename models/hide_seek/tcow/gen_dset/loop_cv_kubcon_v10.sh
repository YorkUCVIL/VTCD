#!/bin/bash
# BVH, Jul 2022.

# conda activate bcv11

for i in {1..45}
do

echo $i

python gen_dset/export_kubcon_v10.py

rm -rf /tmp/*

done

# pkill -9 python
