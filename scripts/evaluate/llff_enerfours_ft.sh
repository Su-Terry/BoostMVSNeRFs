#!/bin/bash

echo > llff_enerfours_ft.txt

scenes=('fern' 'flower' 'fortress' 'horns' 'leaves' 'orchids' 'room' 'trex')
for scene in "${scenes[@]}"
do
    python run.py --type evaluate --cfg_file configs/enerf_ours_ft/llff/${scene}.yaml >> llff_enerfours_ft.txt
done