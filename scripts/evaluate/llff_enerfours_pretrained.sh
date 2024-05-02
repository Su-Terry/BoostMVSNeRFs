#!/bin/bash

echo > llff_enerfours_pretrained.txt

python run.py --type evaluate --cfg_file configs/enerf/llff_eval_ours.yaml >> llff_enerfours_pretrained.txt