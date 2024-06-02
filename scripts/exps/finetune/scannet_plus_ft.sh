#!/bin/bash

METHOD="$1"
SCENE="$2"

SCANNET_SCENES=('scene0000_01' 'scene0079_00' 'scene0158_00' 'scene0316_00' 'scene0521_00' 'scene0553_00' 'scene0616_00' 'scene0653_00')

# check if the scene is in the list of free scenes
for var in "${SCANNET_SCENES[@]}"; do
    if [[ $var == $SCENE ]]; then
        python train_net.py --cfg_file "configs/exps/finetune/$METHOD/scannet_plus/${SCENE}.yaml"
        exit 0
    fi
done

echo "Invalid scene name. Please choose from: ${FREE_SCENES[@]}"
exit 1