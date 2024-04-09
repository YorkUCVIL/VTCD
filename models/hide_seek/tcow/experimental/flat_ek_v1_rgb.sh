#!/bin/bash

for j in {01..32}
do
for i in {01..32}
do

echo $j $i

ln -s /proj/vondrick/datasets/epic-kitchens/v1/data/raw/rgb/P${j}/P${j}_${i} /proj/vondrick/datasets/epic-kitchens/v1/data/raw/rgb_flat/P${j}_${i}

done
done
