#! /bin/bash

#SBATCH -N 1
#SBATCH -c 1
#SBATCH --gres=gpu
#SBATCH -p ug-gpu-small
#SBATCH --qos=short
#SBATCH --job-name=TestAll
#SBATCH --mem=12g
#SBATCH -t 02-00:00:00

source /etc/profile
source env/bin/activate

python3 test_model.py experiments_v4_dcvnt_noaug_imp10_infmask0010_t2_1pc NpyData-size224x336-imp10-infmask0010-test-1pc
python3 test_model.py experiments_v4_dcvnt_noaug_imp10_infmask0010_t3_1pc NpyData-size224x336-imp10-infmask0010-test-1pc
python3 test_model.py experiments_v4_dcvnt_noaug_imp10_infmask0010_t4_1pc NpyData-size224x336-imp10-infmask0010-test-1pc
python3 test_model.py experiments_v4_dcvnt_noaug_imp10_infmask0010_t5_1pc NpyData-size224x336-imp10-infmask0010-test-1pc
python3 test_model.py experiments_v4_dcvnt_noaug_infmask2510_t2_1pc NpyData-size224x336-imp25-infmask1010-test-1pc
python3 test_model.py experiments_v4_dcvnt_noaug_infmask5010_t2_1pc NpyData-size224x336-imp50-infmask1010-test-1pc
python3 test_model.py experiments_v4_dcvnt_noaug_infmask1010_t2_noround NpyData-size224x336-infmask1010-test-noround
python3 test_model.py experiments_v4_dcvnt_noaug_infmask1010_t2_1halfpc NpyData-size224x336-infmask1010-test-1halfpc
python3 test_model.py experiments_v4_dcvnt_noaug_infmask1010_t2_2halfpc NpyData-size224x336-infmask1010-test-2halfpc
python3 test_model.py experiments_v4_dcvnt_noaug_infmask1010_t2_5pc NpyData-size224x336-infmask1010-test-5pc
python3 test_model.py experiments_v4_dcvnt_noaug_infmask1010_t2_5pc_5ep NpyData-size224x336-infmask1010-test-5pc
python3 test_model.py experiments_v4_dcvnt_noaug_infmask1010_t2 NpyData-size224x336-infmask1010-test2
python3 test_model.py experiments_v4_dcvnt_noaug_infmask1010_t2_1pc NpyData-size224x336-infmask1010-test-1pc
python3 test_model.py experiments_v4_dcvnt_noaug_impdlmask_run2 NpyData-size224x336-impdlmask
