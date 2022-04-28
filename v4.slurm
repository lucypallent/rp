#!/bin/bash

#SBATCH -N 1
#SBATCH -c 1
#SBATCH --gres=gpu
#SBATCH -p ug-gpu-small
#SBATCH --qos=short
#SBATCH --job-name=T3Cmbned
# imp10T3
# imp10T4
# Timp10
#SBATCH --mem=12g
#SBATCH -t 02-00:00:00

source /etc/profile
source env/bin/activate

python3 test_model_combined.py experiments_v4_dcvnt_noaug_imp10_infmask0010_t3_1pc NpyData-size224x336-imp10-infmask0010-test-1pc
# python3 dcvnt_v4_dcvnt_noaug_imp10_impdlmasked0010_1pc_t3_covcap_ft.py
# python3 dcvnt_v4_dcvnt_noaug_imp10_impdlmasked0010_1pc_t3.py
# python3 dcvnt_v4_dcvnt_noaug_imp10_impdlmasked0010_1pc_t4.py
