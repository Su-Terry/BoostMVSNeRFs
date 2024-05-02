#!/bin/bash

scenes=('fern' 'flower' 'fortress' 'horns' 'leaves' 'orchids' 'room' 'trex')
for scene in "${scenes[@]}"
do
    python train_net.py --cfg_file configs/enerf_ours_ft/llff/${scene}.yaml
done