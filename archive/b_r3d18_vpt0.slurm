#!/bin/bash

#SBATCH -N 1
#SBATCH -c 1
#SBATCH --gres=gpu
#SBATCH -p ug-gpu-small
#SBATCH --qos=short
#SBATCH --job-name=lpTest
#SBATCH --mem=8g
#SBATCH -t 02:00:00

source /etc/profile
source env/bin/activate

python3 b_r3d18_vpt0.py
