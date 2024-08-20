#!/bin/bash

# Writing to /tmp directory of the node (faster, can mount to data/ziz afterwards)
#SBATCH --output=/tmp/slurm-%j.out
#SBATCH --error=/tmp/slurm-%j.out

# Name of job
#SBATCH --job-name=test_edm

# Using thet cluster srf_gpu_01 and node 6
#SBATCH --cluster=srf_gpu_01
#SBATCH --partition=high-bigbayes-test
#SBATCH --gres=gpu:1

# Change if you know what doing (look at examples, notes)
#SBATCH --cpus-per-task=8

# This is useful for selecting the particular nodes that you want
#NOTSBATCH --nodelist=zizgpu06.cpu.stats.ox.ac.uk

# Make sure RAM Is enough otherwise it will crash
#SBATCH --time=00-12:00:00  
#SBATCH --mem=32G  

# Don't change unless you know why (look at examples and notes for more information)
#SBATCH --ntasks=1

echo "bruh"

export WANDB_API_KEY="78d61cd721affd9ffa2f5e217ed6f49de71eb842"

export PATH_TO_CONDA="/data/localhost/not-backed-up/$USER/miniconda3"
source $PATH_TO_CONDA/bin/activate symdiff_env

# Create central and local logging directories
export LOCAL_LOGDIR="/data/localhost/not-backed-up/$USER/symdiff"
mkdir -p $LOCAL_LOGDIR/output
mkdir -p $LOCAL_LOGDIR/slurm

# The central logdir could be in /data/ziz/$USER or /data/ziz/not-backed-up/$USER instead of /data/ziz/not-backed-up/scratch/
export CENTRAL_LOGDIR="/data/ziz/not-backed-up/$USER/projects/symdiff"
mkdir -p $CENTRAL_LOGDIR
mkdir -p $CENTRAL_LOGDIR/outputs
mkdir -p $CENTRAL_LOGDIR/outputs/slurm

echo "SLURM_JOBID: " $SLURM_JOBID
echo "bruh"
date -u

#python main_qm9.py --n_epochs 1 --exp_name test_no_h --n_stability_samples 1000 --diffusion_noise_schedule polynomial_2 \
#       --diffusion_noise_precision 1e-5 --diffusion_steps 1000 --diffusion_loss_type l2 --batch_size 64 --nf 128 --n_layers 6 --lr 1e-4 --normalize_factors [1,4,10] \
#        --test_epochs 20 --ema_decay 0.9999 --wandb_usr zhangleo1209 --dataset qm9  --datadir /data/zizgpu06/not-backed-up/nvme00/lezhang \
#        --save_model True --filter_n_atoms 4
python sym_nn/sym_nn.py
date -u

# script to run main.py
echo "Job completed."
#cp /tmp/slurm-${SLURM_JOB_ID}.out ${LOCAL_LOGDIR}/slurm/.
cp /tmp/slurm-${SLURM_JOB_ID}.out ${CENTRAL_LOGDIR}/outputs/slurm/.
#cp -r $LOCAL_LOGDIR/output/* $CENTRAL_LOGDIR/output/.