#!/bin/bash
#SBATCH -J lignting
#SBATCH -p gpu
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --mem=32G

#SBATCH --gres=gpu:1
#SBATCH -o outpastes.%j
#SBATCH -e err.%j
#SBATCH -t 1200:00:00


PYTHON=/public/home/jd_cdy/anaconda3/envs/pytorch/bin/python
cd  $SLURM_SUBMIT_DIR
NP=$SLURM_JOB_GPUS

CUDA_VISIBLE_DEVICES=$NP 
$PYTHON train.py 

