#!/bin/bash

scenes=('scene0000_01' 'scene0079_00' 'scene0158_00' 'scene0316_00' 'scene0521_00' 'scene0553_00' 'scene0616_00' 'scene0653_00')
for scene in "${scenes[@]}"
do
    python train_net.py --cfg_file configs/mvsnerf_ours/scannet/${scene}_ft.yaml
done