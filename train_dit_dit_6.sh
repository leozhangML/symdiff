#!/bin/bash
# Writing to /tmp directory of the node (faster, can mount to data/ziz afterwards)
#SBATCH --output=/tmp/slurm-%j.out
#SBATCH --error=/tmp/slurm-%j.out

# Name of job
#SBATCH --job-name=DiT_DiT_noise_big

# Using thet cluster srf_gpu_01 and node 6
#SBATCH --cluster=srf_gpu_01
#SBATCH --partition=high-bigbayes-test
#SBATCH --gres=gpu:1

# Change if you know what doing (look at examples, notes)
#SBATCH --cpus-per-task=4

# This is useful for selecting the particular nodes that you want
#NOTSBATCH --nodelist=zizgpu06.cpu.stats.ox.ac.uk

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

python main_qm9.py --exp_name DiT_DiT_noise_big --model dit_dit_dynamics --dataset qm9  --datadir /data/zizgpu06/not-backed-up/nvme00/lezhang \
                   --diffusion_noise_precision 1e-5 --diffusion_steps 1000 --diffusion_loss_type l2 --diffusion_noise_schedule polynomial_2 \
                   --n_epochs 3000 --batch_size 256 --lr 1e-4 --com_free --clipping_type norm --max_grad_norm 2.0 --ema_decay 0.999 \
                   --weight_decay 0.00001 --use_amsgrad --normalize_factors [1,4,10] \
                   --n_stability_samples 500 --test_epochs 20 --wandb_usr zhangleo1209 --save_model True \
                   --enc_out_channels 0 --enc_x_scale 1.0 --enc_hidden_size 128 --enc_depth 4 --enc_num_heads 8 --enc_mlp_ratio 1.0 \
                   --dec_hidden_features 64 --enc_x_emb linear \
                   --x_scale 1.0 --hidden_size 512 --depth 6 --num_heads 8 --mlp_ratio 2.0 --x_emb linear \
                   --noise_dims 3 --noise_std 1.0 \
                   --print_grad_norms --print_parameter_count
date -u

# script to run main.py
echo "Job completed."
#cp /tmp/slurm-${SLURM_JOB_ID}.out ${LOCAL_LOGDIR}/slurm/.
cp /tmp/slurm-${SLURM_JOB_ID}.out ${CENTRAL_LOGDIR}/outputs/slurm/.
#cp -r $LOCAL_LOGDIR/output/* $CENTRAL_LOGDIR/output/.