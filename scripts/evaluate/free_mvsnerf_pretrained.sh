#!/bin/bash

echo > free_mvsnerf_pretrained.txt

python run.py --type evaluate --cfg_file configs/mvsnerf/free_eval.yaml >> free_mvsnerf_pretrained.txt