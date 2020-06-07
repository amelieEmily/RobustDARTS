#!/bin/bash

#SBATCH --mail-type=ALL
#SBATCH --mail-user=ys3316

export PATH=/vol/bitbucket/${USER}/miniconda3/bin/:$PATH
source activate
source /vol/cuda/10.0.130/setup.sh
TERM=vt100 # or TERM=xterm
/usr/bin/nvidia-smi
uptime

python src/search/randomNAS/random_weight_share.py --seed 10 --epochs 50 --save experiments/search_logs_RandomNAS --space s3 --dataset cifar10 --drop_path_prob 0 --weight_decay 0.0003
