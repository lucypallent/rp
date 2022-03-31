#!/bin/bash

#SBATCH -N 1
#SBATCH -c 1
#SBATCH --gres=gpu
#SBATCH -p ug-gpu-small
#SBATCH --qos=debug
#SBATCH --job-name=lpTest
#SBATCH --mem=8g

source /etc/profile
source env/bin/activate

python3 b_r3d18_v0_t.py
