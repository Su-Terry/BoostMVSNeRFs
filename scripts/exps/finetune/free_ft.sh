#!/bin/bash

METHOD="$1"
SCENE="$2"

FREE_SCENES=('grass' 'hydrant' 'lab' 'pillar' 'road' 'sky' 'stair')

if [[ " ${FREE_SCENES[*]} " =~ [[:space:]]${SCENE}[[:space:]] ]]; then
    python train_net.py --cfg_file "configs/exps/finetune/$Method/free/${SCENE}_ft.yaml"
else
    echo "Invalid scene name. Please choose from: ${FREE_SCENES[@]}"
    exit 1
fi