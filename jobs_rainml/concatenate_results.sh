#!/bin/bash

# Emails
#SBATCH --mail-user=kianoosh.ashouritaklimi@stats.ox.ac.uk
#SBATCH --mail-type=ALL

# Writing to /tmp directory of the node (faster, can mount to data/ziz afterwards)
#SBATCH --output=/tmp/slurm-%j.out
#SBATCH --error=/tmp/slurm-%j.out

# Name of job
#SBATCH --job-name=process_evals

# Using the cluster swan and node 1 
#SBATCH --cluster=srf_gpu_01
#SBATCH --partition=high-bigbayes-test
#SBATCH --gres=gpu:1

# Change if you know what doing (look at examples, notes)
#SBATCH --cpus-per-task=1

# This is useful for selecting the particular nodes that you want
#NOTSBATCH --nodelist=zizgpu05.cpu.stats.ox.ac.uk


# Make sure RAM is enough
#SBATCH --time=12-00:00:00  
#SBATCH --mem=42G  

# Don't change unless you know why (look at examples and notes for more information)
#SBATCH --ntasks=1



# Ensure the script stops if any command fails
set -e

# Function to display usage information
usage() {
    echo "Usage: $0 --experiment_names <exp1 exp2 ...> --directory_path_1 <path> --directory_path_2 <path>"
    exit 1
}

# Parse the arguments
while [[ "$#" -gt 0 ]]; do
    case "$1" in
        --experiment_names)
            shift
            experiment_names=()
            while [[ "$#" -gt 0 && "$1" != --* ]]; do
                experiment_names+=("$1")
                shift
            done
            ;;
        --directory_path_1)
            directory_path_1="$2"
            shift 2
            ;;
        --directory_path_2)
            directory_path_2="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            usage
            ;;
    esac
done

# Ensure all required arguments are provided
if [[ -z "${experiment_names[*]}" || -z "$directory_path_1" || -z "$directory_path_2" ]]; then
    usage
fi

# # Define the output file
# output_file="$directory_path_1/$experiment_name/all_experiment_eval_logs.txt"

# # Create or overwrite the output file
# echo "Creating $output_file"
# > "$output_file"

# Loop through each experiment name
for experiment_name in "${experiment_names[@]}"; do
    # Define the output file
    output_file="$directory_path_1/$experiment_name/all_experiment_eval_logs.txt"

    # Create or overwrite the output file
    echo "Creating $output_file"
    > "$output_file"
    
    experiment_dir="$directory_path_1/$experiment_name"
    for j in {0..2}; do

        all_evals="$experiment_dir/all_evals_seed_$j.txt"

        # Ensure the experiment directory exists
        if [[ ! -d "$experiment_dir" ]]; then
            echo "Experiment directory $experiment_dir does not exist, skipping..."
            continue
        fi

        # Ensure the all_evals.txt file exists
        if [[ ! -f "$all_evals" ]]; then
            echo "all_evals.txt not found in $experiment_dir, skipping..."
            continue
        fi

        # Append the experiment name to the output file
        echo "Adding logs for experiment: $experiment_name"
        echo "$experiment_name" >> "$output_file"

        # Append the contents of all_evals.txt to the output file
        cat "$all_evals" >> "$output_file"

        # Append a separator line to the output file
        echo "--------------------------------" >> "$output_file"
    done
done

echo "All logs have been consolidated into $output_file"


# scp -r the all_experiment_eval_logs.txt file to zizgpu04.cpu.stats.ox.ac.uk:/data/localhost/not-backed-up/ashouritaklimi/realistic_epig/results/
# scp -r $output_file zizgpu04.cpu.stats.ox.ac.uk:/data/localhost/not-backed-up/ashouritaklimi/symdiff