#!/bin/bash

#SBATCH --mail-type=ALL
#SBATCH --mail-user=ys3316

python src/search/train_search.py --unrolled --train_portion 0.1 --cutout --report_freq_hessian 2 --space s1 --dataset dr-detection --drop_path_prob 0.2 --weight_decay 0.0121

#datasets="dr-detection"
#spaces="s1 s2 s3 s4"
#wdecay=$(awk 'BEGIN{for(i=0.0003;i<=0.0243;i*=7)print i}')
#dpath=$(awk 'BEGIN{for(j=0.0;j<=0.6;j+=0.2)print j}')
#
#
#for d in $datasets; do
#  for s in $spaces; do
#		sbatch scripts/DARTS_search.sh $s $d 0.0121 0.4
#		echo submmited job $s $d 0.0121 0.4
#  done
#done

