#!/bin/bash
#SBATCH --job-name=test_edm
#SBATCH --partition=high-bigbayes-gpu
#SBATCH -M srf_gpu_01
#SBATCH --nodelist=zizgpu06.cpu.stats.ox.ac.uk
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=03:00:00
#SBATCH --mem-per-cpu=6G
#SBATCH --output=/data/localhost/not-backed-up/lezhang/jobname_%j.txt

# script to run main.py

source ~/miniconda3/bin/activate

conda info --envs
conda deactivate
conda activate symdiff_env

nvidia-smi

#jobname="test"

mkdir -p /data/ziz/not-backed-up/lezhang/results/jobname_${SLURM_JOBID}
out_dir="/data/ziz/not-backed-up/lezhang/results/jobname_$SLURM_JOBID"
log_file="/data/ziz/not-backed-up/lezhang/results/jobname_$SLURM_JOBID/log.log"

echo $out_dir

#print environment variables: the job ID.
echo "SLURM_JOBID: " $SLURM_JOBID
echo "bruh"
date -u

# run file

python main_qm9.py --n_epochs 3000 --exp_name edm_qm9 --n_stability_samples 1000 --diffusion_noise_schedule polynomial_2 \
       --diffusion_noise_precision 1e-5 --diffusion_steps 1000 --diffusion_loss_type l2 --batch_size 64 --nf 256 --n_layers 9 --lr 1e-4 --normalize_factors [1,4,10] \
        --test_epochs 20 --ema_decay 0.9999

date -u

# Make a job directory for output and results if it does not exist
mv /data/localhost/not-backed-up/lezhang/jobname_${SLURM_JOBID}.txt /data/ziz/not-backed-up/lezhang/results/jobname_${SLURM_JOBID}/jobname_${SLURM_JOBID}.txt
