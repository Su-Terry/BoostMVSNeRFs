#!/bin/bash

METHOD="$1"
SCENE="$2"

SCANNET_SCENES=('scene0000_01' 'scene0079_00' 'scene0158_00' 'scene0316_00' 'scene0521_00' 'scene0553_00' 'scene0616_00' 'scene0653_00')

if [[ " ${SCANNET_SCENES[*]} " =~ [[:space:]]${SCENE}[[:space:]] ]]; then
    python run.py --type evaluate --cfg_file "configs/exps/finetune/$Method/scannet_plus/${SCENE}_ft.yaml"
else
    echo "Invalid scene name. Please choose from: ${SCANNET_SCENES[@]}"
    exit 1
fi