#!/bin/bash

echo > scannet_enerf_pretrained.txt

python run.py --type evaluate --cfg_file configs/enerf/scannet_eval.yaml >> scannet_enerf_pretrained.txt