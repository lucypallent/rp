#!/bin/bash

#SBATCH -N 1
#SBATCH -c 1
#  #SBATCH --gres=gpu
#SBATCH -p cpu
#SBATCH --qos=short
#SBATCH --job-name=2ArbMLP
#SBATCH -t 00-20:00:00

source /etc/profile
source env/bin/activate

python3 2A-rob-MLP.py
