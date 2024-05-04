#!/bin/bash

scenes=('grass' 'hydrant' 'lab' 'pillar' 'road' 'sky' 'stair')
for scene in "${scenes[@]}"
do
    python train_net.py --cfg_file configs/enerf_ours_ft/free/${scene}_ft.yaml
done