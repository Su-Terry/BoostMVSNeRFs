#!/bin/bash

echo > scannet_enerf_ft.txt

scenes=('scene0000_01' 'scene0079_00' 'scene0158_00' 'scene0316_00' 'scene0521_00' 'scene0553_00' 'scene0616_00' 'scene0653_00')
for scene in "${scenes[@]}"
do
    python run.py --type mcp --cfg_file configs/enerf_ours_ft/scannet/${scene}_ft.yaml
done