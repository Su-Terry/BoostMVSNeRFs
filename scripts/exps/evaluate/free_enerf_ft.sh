#!/bin/bash

echo > free_enerf_ft.txt

scenes=('grass' 'hydrant' 'lab' 'pillar' 'road' 'sky' 'stair')
for scene in "${scenes[@]}"
do
    python run.py --type evaluate --cfg_file configs/enerf_ft/free_eval/${scene}_ft_enerf.yaml >> free_enerf_ft.txt
done