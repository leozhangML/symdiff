#!/bin/bash

# Emails
#SBATCH --mail-user=kianoosh.ashouritaklimi@stats.ox.ac.uk
#SBATCH --mail-type=ALL

# Writing to /tmp directory of the node (faster, can mount to data/ziz afterwards)
#SBATCH --output=/tmp/slurm-%j.out
#SBATCH --error=/tmp/slurm-%j.out

# Name of job
#SBATCH --job-name=best_optimisation_ddg_lr_1e_4_5m

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

# Run main script
# Arguments ignored: --enc_concat_h
python /data/localhost/not-backed-up/users/$USER/symdiff/main_geom_drugs.py --exp_name best_optimisation_ddg_lr_1e_4_5m --model dit_dit_gaussian_dynamics --dataset geom --datadir /data/localhost/not-backed-up/users/$USER/data/geom/geom_drugs_30.npy \
                   --diffusion_noise_precision 1e-5 --diffusion_steps 1000 --diffusion_loss_type l2 --diffusion_noise_schedule polynomial_2 \
                   --n_epochs 40 --batch_size 256 --lr 1e-4 --com_free --clipping_type norm --max_grad_norm 2.0 --ema_decay 0.9999 \
                   --weight_decay 1e-12 --use_amsgrad --normalize_factors [1,4,10] \
                   --n_stability_samples 500 --test_epochs 5 --wandb_usr kiaashouri --save_model True \
                   --xh_hidden_size 184 --K 184 \
                   --mlp_type swiglu \
                   --enc_hidden_size 128 --enc_depth 4 --enc_num_heads 4 --enc_mlp_ratio 4.0 --dec_hidden_features 64 \
                   --hidden_size 228 --depth 6 --num_heads 6 --mlp_ratio 4.0 --mlp_dropout 0.0 \
                   --noise_dims 3 --noise_std 1.0 \
                   --print_parameter_count 


date -u
echo "Job completed."
cp /tmp/slurm-${SLURM_JOB_ID}.out ${LOCAL_LOGDIR}/slurm/.