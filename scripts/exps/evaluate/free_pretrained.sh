#!/bin/bash

METHOD="$1"

python run.py --type evaluate --cfg_file "configs/exps/evaluate/$METHOD/free_eval.yaml"