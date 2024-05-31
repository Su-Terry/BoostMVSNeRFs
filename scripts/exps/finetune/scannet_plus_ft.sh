#!/bin/bash

METHOD="$1"
SCENE="$2"

SCANNET_SCENES=('grass' 'hydrant' 'lab' 'pillar' 'road' 'sky' 'stair')

# check if the scene is in the list of free scenes
for var in "${SCANNET_SCENES[@]}"; do
    if [[ $var == $SCENE ]]; then
        python train_net.py --cfg_file "configs/exps/finetune/$METHOD/scannet_plus/${SCENE}.yaml"
        exit 0
    fi
done

echo "Invalid scene name. Please choose from: ${FREE_SCENES[@]}"
exit 1