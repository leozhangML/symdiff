#!/bin/bash

# Writing to /tmp directory of the node (faster, can mount to data/ziz afterwards)
#SBATCH --output=/tmp/slurm-%j.out
#SBATCH --error=/tmp/slurm-%j.out

# Emails
#SBATCH --mail-user=kianoosh.ashouritaklimi@stats.ox.ac.uk
#SBATCH --mail-type=ALL

# Name of job
#SBATCH --job-name=eval_best_optimisation_ddg_lr_1e_4_10m_seed_1

# Using thet cluster swan and node 1 
#SBATCH --cluster=swan
#SBATCH --partition=standard-rainml-gpu
#SBATCH --gres=gpu:Ampere_H100_80GB:1

# Change if you know what doing (look at examples, notes)
#SBATCH --cpus-per-task=1

# This is useful for selecting the particular nodes that you want
#NOTSBATCH --nodelist=rainmlgpu01.cpu.stats.ox.ac.uk


# Make sure RAM is enough
#SBATCH --time=12-00:00:00  
#SBATCH --mem=42G  

# Don't change unless you know why (look at examples and notes for more information)
#SBATCH --ntasks=1

export WANDB_API_KEY="b251e22c374eb363aa6189f343b015bbbeb6bad7"
export PATH_TO_CONDA="/data/localhost/not-backed-up/users/$USER/miniconda3"
source $PATH_TO_CONDA/bin/activate symdiff

# Create central and local logging directories
export LOCAL_LOGDIR="/data/localhost/not-backed-up/users/$USER/symdiff"
mkdir -p $LOCAL_LOGDIR/output
mkdir -p $LOCAL_LOGDIR/slurm


echo "SLURM_JOBID: " $SLURM_JOBID
date -u

#python eval_analyze.py --model_path outputs/edm_9_4_m --n_samples 10000 --datadir /data/zizgpu06/not-backed-up/nvme00/lezhang  --data_aug_at_sampling
cd /data/localhost/not-backed-up/users/$USER/symdiff
python /data/localhost/not-backed-up/users/$USER/symdiff/eval_analyze.py --model_path outputs/best_optimisation_ddg_lr_1e_4_10m_seed_1 --n_samples 10000 --datadir /data/localhost/not-backed-up/users/$USER/data/geom/geom_drugs_30.npy


date -u
echo "Job completed."
cp /tmp/slurm-${SLURM_JOB_ID}.out ${LOCAL_LOGDIR}/slurm/.