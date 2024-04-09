#!/bin/bash

for j in {01..50}
do
for i in {101..150}
do

echo $j $i

ln -s /proj/vondrick/datasets/epic-kitchens/v2/data/P${j}/rgb_frames/P${j}_${i} /proj/vondrick/datasets/epic-kitchens/v2/data/rgb_flat/P${j}_${i}

done
done
