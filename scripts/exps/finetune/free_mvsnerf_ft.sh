#!/bin/bash

scenes=('grass' 'hydrant' 'lab' 'pillar' 'road' 'sky' 'stair')
for scene in "${scenes[@]}"
do
    python train_net.py --cfg_file configs/mvsnerf/free/${scene}_ft.yaml
done