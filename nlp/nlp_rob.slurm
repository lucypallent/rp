#!/bin/bash

#SBATCH -N 1
#SBATCH -c 1
#############SBATCH --gres=gpu
#SBATCH -p cpu
#SBATCH --qos=short
#SBATCH --job-name=creRob
#SBATCH --mem=60g
#SBATCH -t 02-00

source /etc/profile
source env/bin/activate

#python3 create_unique.py

#python3 create_unique2-distilrobert.py

python3 create_roberta.py
