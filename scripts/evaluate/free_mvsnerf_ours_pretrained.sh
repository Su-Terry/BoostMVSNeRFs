#!/bin/bash

echo > free_mvsnerf_ours_pretrained.txt

python run.py --type evaluate --cfg_file configs/mvsnerf_ours/free_eval.yaml >> free_mvsnerf_ours_pretrained.txt