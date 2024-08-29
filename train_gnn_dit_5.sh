#!/bin/bash

# Writing to /tmp directory of the node (faster, can mount to data/ziz afterwards)
#SBATCH --output=/tmp/slurm-%j.out
#SBATCH --error=/tmp/slurm-%j.out

# Name of job
#SBATCH --job-name=GNN_DiT

# Using thet cluster srf_gpu_01 and node 5
#SBATCH --cluster=srf_gpu_01
#SBATCH --partition=high-bigbayes-gpu
#SBATCH --gres=gpu:1

# Change if you know what doing (look at examples, notes)
#SBATCH --cpus-per-task=4

# This is useful for selecting the particular nodes that you want
#SBATCH --nodelist=zizgpu05.cpu.stats.ox.ac.uk

# Make sure RAM Is enough otherwise it will crash
#SBATCH --time=12-00:00:00  
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

python main_qm9.py --n_epochs 2500 --exp_name GNN_DiT --model gnn_dit_dynamics --n_stability_samples 500 --diffusion_noise_schedule polynomial_2 \
       --diffusion_noise_precision 1e-5 --diffusion_steps 1000 --diffusion_loss_type l2 --batch_size 256 --lr 1e-4 --normalize_factors [1,4,10] \
        --test_epochs 20 --ema_decay 0.9999 --wandb_usr zhangleo1209 --dataset qm9  --datadir /data/zizgpu05/not-backed-up/nvme00/lezhang \
        --x_scale 25.0 --hidden_size 256 --depth 8 --num_heads 8 --mlp_ratio 1.5 --com_free --use_amsgrad --clipping_type norm --max_grad_norm 1.5 \
        --gamma_gnn_layers 4 --gamma_gnn_hidden_size 64 --gamma_gnn_out_size 64 --gamma_dec_hidden_size 32 --x_emb linear

date -u

# script to run main.py
echo "Job completed."
#cp /tmp/slurm-${SLURM_JOB_ID}.out ${LOCAL_LOGDIR}/slurm/.
cp /tmp/slurm-${SLURM_JOB_ID}.out ${CENTRAL_LOGDIR}/outputs/slurm/.
#cp -r $LOCAL_LOGDIR/output/* $CENTRAL_LOGDIR/output/.