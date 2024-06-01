#!/bin/bash

METHOD="$1"

python run.py --type evaluate --cfg_file "configs/exps/evaluate/$METHOD/scannet_plus_eval.yaml"