#!/bin/bash

echo > scannet_enerfours_pretrained.txt

python run.py --type evaluate --cfg_file configs/enerf/scannet_eval_ours.yaml >> scannet_enerfours_pretrained.txt
