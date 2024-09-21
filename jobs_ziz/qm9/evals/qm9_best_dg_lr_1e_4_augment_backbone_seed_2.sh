#!/bin/bash

# Emails
#SBATCH --mail-user=kianoosh.ashouritaklimi@stats.ox.ac.uk
#SBATCH --mail-type=ALL

# Writing to /tmp directory of the node (faster, can mount to data/ziz afterwards)
#SBATCH --output=/tmp/slurm-%j.out
#SBATCH --error=/tmp/slurm-%j.out

# Name of job
#SBATCH --job-name=qm9_best_dg_lr_1e_4_augment_backbone_seed_2

# Using thet cluster swan and node 1 
#SBATCH --cluster=srf_gpu_01
#SBATCH --partition=high-bigbayes-gpu
#SBATCH --gres=gpu:1

# Change if you know what doing (look at examples, notes)
#SBATCH --cpus-per-task=1

# This is useful for selecting the particular nodes that you want
#SBATCH --nodelist=zizgpu05.cpu.stats.ox.ac.uk


# Make sure RAM is enough
#SBATCH --time=12-00:00:00  
#SBATCH --mem=42G  

# Don't change unless you know why (look at examples and notes for more information)
#SBATCH --ntasks=1

export WANDB_API_KEY="b251e22c374eb363aa6189f343b015bbbeb6bad7"
export PATH_TO_CONDA="/data/localhost/not-backed-up/$USER/miniforge3"
source $PATH_TO_CONDA/bin/activate symdif

# Create central and local logging directories
export LOCAL_LOGDIR="/data/localhost/not-backed-up/$USER/symdiff"
mkdir -p $LOCAL_LOGDIR/output
mkdir -p $LOCAL_LOGDIR/slurm


echo "SLURM_JOBID: " $SLURM_JOBID
date -u

#python eval_analyze.py --model_path outputs/edm_9_4_m --n_samples 10000 --datadir /data/localhost/not-backed-up/nvme00/ashouritaklimi  --data_aug_at_sampling
cd /data/localhost/not-backed-up/$USER/symdiff
python /data/localhost/not-backed-up/$USER/symdiff/eval_analyze.py --model_path outputs/qm9_best_dg_lr_1e_4_augment_backbone_seed_2 --n_samples 10000 --datadir /data/localhost/not-backed-up/nvme00/ashouritaklimi


date -u
echo "Job completed."
cp /tmp/slurm-${SLURM_JOB_ID}.out ${LOCAL_LOGDIR}/slurm/.