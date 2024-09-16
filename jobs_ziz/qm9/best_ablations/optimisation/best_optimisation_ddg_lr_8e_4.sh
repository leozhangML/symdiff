#!/bin/bash

# Emails
#SBATCH --mail-user=kianoosh.ashouritaklimi@stats.ox.ac.uk
#SBATCH --mail-type=ALL

# Writing to /tmp directory of the node (faster, can mount to data/ziz afterwards)
#SBATCH --output=/tmp/slurm-%j.out
#SBATCH --error=/tmp/slurm-%j.out

# Name of job
#SBATCH --job-name=qm9_best_ddg_lr_8e_4

# Using the cluster swan and node 1 
#SBATCH --cluster=srf_gpu_01
#SBATCH --partition=high-bigbayes-test
#SBATCH --gres=gpu:1

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
export PATH_TO_CONDA="/data/localhost/not-backed-up/$USER/miniforge3"
source $PATH_TO_CONDA/bin/activate symdif

# Create central and local logging directories
export LOCAL_LOGDIR="/data/localhost/not-backed-up/$USER/symdiff"
mkdir -p $LOCAL_LOGDIR/output
mkdir -p $LOCAL_LOGDIR/slurm

# The central logdir could be in /data/ziz/$USER or /data/ziz/not-backed-up/$USER instead of /data/ziz/not-backed-up/scratch/
export CENTRAL_LOGDIR="/data/ziz/not-backed-up/$USER/symdiff"
mkdir -p $CENTRAL_LOGDIR
mkdir -p $CENTRAL_LOGDIR/outputs
mkdir -p $CENTRAL_LOGDIR/outputs/slurm


echo "SLURM_JOBID: " $SLURM_JOBID
date -u


# Run main script
# Arguments ignored: --enc_concat_h
cd /data/localhost/not-backed-up/$USER/symdiff
python /data/localhost/not-backed-up/$USER/symdiff/main_qm9.py --exp_name qm9_best_ddg_lr_8e_4 --model dit_dit_gaussian_dynamics --dataset qm9 --datadir /data/localhost/not-backed-up/nvme00/ashouritaklimi \
                   --diffusion_noise_precision 1e-5 --diffusion_steps 1000 --diffusion_loss_type l2 --diffusion_noise_schedule polynomial_2 \
                   --n_epochs 5000 --batch_size 256 --lr 8e-4 --com_free --clipping_type norm --max_grad_norm 2.0 --ema_decay 0.9999 \
                   --weight_decay 1e-12 --use_amsgrad --normalize_factors [1,4,10] \
                   --n_stability_samples 500 --test_epochs 20 --wandb_usr kiaashouri --save_model True \
                   --xh_hidden_size 184 --K 184 \
                   --mlp_type swiglu \
                   --enc_hidden_size 128 --enc_depth 8 --enc_num_heads 4 --enc_mlp_ratio 4.0 --dec_hidden_features 64 \
                   --hidden_size 384 --depth 12 --num_heads 6 --mlp_ratio 4.0 --mlp_dropout 0.0 \
                   --noise_dims 3 --noise_std 1.0 \
                   --print_parameter_count 


date -u
echo "Job completed."
cp /tmp/slurm-${SLURM_JOB_ID}.out ${LOCAL_LOGDIR}/slurm/.