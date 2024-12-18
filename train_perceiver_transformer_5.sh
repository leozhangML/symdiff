#!/bin/bash

# Writing to /tmp directory of the node (faster, can mount to data/ziz afterwards)
#SBATCH --output=/tmp/slurm-%j.out
#SBATCH --error=/tmp/slurm-%j.out

# Name of job
#SBATCH --job-name=perceiver_transformer

# Using thet cluster srf_gpu_01 and node 5
#SBATCH --cluster=srf_gpu_01
#SBATCH --partition=high-bigbayes-gpu
#SBATCH --gres=gpu:1

# Change if you know what doing (look at examples, notes)
#SBATCH --cpus-per-task=12

# This is useful for selecting the particular nodes that you want
#SBATCH --nodelist=zizgpu05.cpu.stats.ox.ac.uk

# Make sure RAM Is enough otherwise it will crash
#SBATCH --time=12-00:00:00  
#SBATCH --mem=80G  

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

python main_qm9.py --n_epochs 2000 --exp_name perceiver_transformer_test --model symdiff_transformer_dynamics --n_stability_samples 500 --diffusion_noise_schedule polynomial_2 \
       --diffusion_noise_precision 1e-5 --diffusion_steps 1000 --diffusion_loss_type l2 --batch_size 512 --lr 1e-4 --normalize_factors [1,4,10] \
        --test_epochs 20 --ema_decay 0.999 --wandb_usr zhangleo1209 --dataset qm9  --datadir /data/zizgpu05/not-backed-up/nvme00/lezhang  \
        --save_model True --gamma_num_enc_layers 2 --gamma_num_dec_layers 2 --gamma_d_model 64 --gamma_nhead 4 --gamma_dim_feedforward 64 \
        --gamma_dropout 0.1 --k_num_layers 10 --k_d_model 256 --k_nhead 8 --k_dim_feedforward 512 --k_dropout 0.1 --num_bands 32 --max_resolution 100 \
        --t_fourier True --concat_t False --com_free True --clip_grad True --scheduler cosine \
        --num_warmup_steps 50000 --num_training_steps 392000 --dp False

date -u

# script to run main.py
echo "Job completed."
#cp /tmp/slurm-${SLURM_JOB_ID}.out ${LOCAL_LOGDIR}/slurm/.
cp /tmp/slurm-${SLURM_JOB_ID}.out ${CENTRAL_LOGDIR}/outputs/slurm/.
#cp -r $LOCAL_LOGDIR/output/* $CENTRAL_LOGDIR/output/.