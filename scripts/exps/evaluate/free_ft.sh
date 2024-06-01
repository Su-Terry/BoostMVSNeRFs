#!/bin/bash

METHOD="$1"
SCENE="$2"

FREE_SCENES=('grass' 'hydrant' 'lab' 'pillar' 'road' 'sky' 'stair')

if [[ " ${FREE_SCENES[*]} " =~ [[:space:]]${SCENE}[[:space:]] ]]; then
    python run.py --type evaluate --cfg_file "configs/exps/finetune/$METHOD/free/${SCENE}.yaml"
else
    echo "Invalid scene name. Please choose from: ${FREE_SCENES[@]}"
    exit 1
fi