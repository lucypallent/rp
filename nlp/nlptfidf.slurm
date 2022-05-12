#!/bin/bash

#SBATCH -N 1
#SBATCH -c 1
#########SBATCH --gres=cpu
#SBATCH -p cpu
#SBATCH --qos=short
#SBATCH --job-name=nlptfR
#SBATCH --mem=60g
#SBATCH -t 02-00

source /etc/profile
source env/bin/activate

python3 nlp-csv.py
