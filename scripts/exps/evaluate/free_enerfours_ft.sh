#!/bin/bash

echo > free_enerfours_ft.txt

scenes=('grass' 'hydrant' 'lab' 'pillar' 'road' 'sky' 'stair')
for scene in "${scenes[@]}"
do
    python run.py --type evaluate --cfg_file configs/enerf_ours_ft/free_eval/${scene}_ft.yaml >> free_enerfours_ft.txt
done