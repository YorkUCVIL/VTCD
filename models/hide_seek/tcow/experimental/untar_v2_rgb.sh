
for j in {01..50}
do
for i in {101..150}
do

echo $j $i

cd /proj/vondrick/datasets/epic-kitchens/v2/data/P${j}/rgb_frames
tar -xf P${j}_${i}.tar --one-top-level

done
done
