#!/bin/bash
#
#PBS -N example16
#PBS -j oe
#PBS -q p72
#PBS -l select=1:ncpus=30:mem=50gb
#PBS -m eb
#PBS -M nikita.leppenen@weizmann.ac.il
#PBS -r n
#PBS -o out_example.txt

#print the time and date of the beginning of the job
date

#print the name of the execution host
echo hostname


eval "$('/apps01/apps/anaconda3-2022.10/bin/conda' 'shell.bash' 'hook' 2> /dev/null)"
conda activate nl97
cd ~
cd /home/nikital/work/projects/quantum_jumps/
python Dicke_with_ind_decay.py
 

#print the time and date at the and of the job
date