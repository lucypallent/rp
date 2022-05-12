#!/bin/bash

#SBATCH -N 1
#SBATCH -c 1
#SBATCH --gres=gpu
#SBATCH -p ug-gpu-small
#SBATCH --qos=short
#SBATCH --job-name=nlp
#SBATCH --mem=28g
#SBATCH -t 02-00

source /etc/profile
source env/bin/activate

python3 nlp-csv.py
