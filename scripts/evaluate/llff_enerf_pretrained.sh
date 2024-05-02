#!/bin/bash

echo > llff_enerf_pretrained.txt

python run.py --type evaluate --cfg_file configs/enerf/llff_eval.yaml >> llff_enerf_pretrained.txt