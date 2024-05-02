#!/bin/bash

echo > llff_enerf_ft.txt

scenes=('fern' 'flower' 'fortress' 'horns' 'leaves' 'orchids' 'room' 'trex')
for scene in "${scenes[@]}"
do
    python run.py --type evaluate --cfg_file configs/enerf_ft/llff/${scene}.yaml >> llff_enerf_ft.txt
done