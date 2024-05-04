#!/bin/bash

echo > free_mvsnerf_ft.txt

scenes=('grass' 'hydrant' 'lab' 'pillar' 'road' 'sky' 'stair')
for scene in "${scenes[@]}"
do
    python run.py --type evaluate --cfg_file configs/mvsnerf/free_eval/${scene}_ft.yaml >> free_mvsnerf_ft.txt
done