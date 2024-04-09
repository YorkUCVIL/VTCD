#!/bin/bash
# BVH, Jul 2022.

# conda activate bcv11

for i in {1..1001}
do

echo $i

python video2frames.py

done

# pkill -9 python
