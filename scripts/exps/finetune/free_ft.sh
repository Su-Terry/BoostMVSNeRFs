#!/bin/bash

METHOD="$1"
SCENE="$2"

FREE_SCENES=('grass' 'hydrant' 'lab' 'pillar' 'road' 'sky' 'stair')

# check if the scene is in the list of free scenes
for var in "${FREE_SCENES[@]}"; do
    if [[ $var == $SCENE ]]; then
        python train_net.py --cfg_file "configs/exps/finetune/$METHOD/free/${SCENE}.yaml"
        exit 0
    fi
done

echo "Invalid scene name. Please choose from: ${FREE_SCENES[@]}"
exit 1