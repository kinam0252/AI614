#!/bin/bash
source /home/nas4_user/kinamkim/.bashrc
conda create -n ai614 python=3.10 -y
conda activate ai614
cd /home/nas4_user/kinamkim/Repos/AI614/workspace
pip install -r requirements.txt
export CUDA_VISIBLE_DEVICES=0
python scripts/run_part1.py

