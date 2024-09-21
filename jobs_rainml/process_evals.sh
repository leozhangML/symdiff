#!/bin/bash

# Emails
#SBATCH --mail-user=kianoosh.ashouritaklimi@stats.ox.ac.uk
#SBATCH --mail-type=ALL

# Writing to /tmp directory of the node (faster, can mount to data/ziz afterwards)
#SBATCH --output=/tmp/slurm-%j.out
#SBATCH --error=/tmp/slurm-%j.out

# Name of job
#SBATCH --job-name=process_evals

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



# Ensure the script stops if any command fails
set -e

# Function to display usage information
usage() {
    echo "Usage: $0 --experiment_names <exp1 exp2 ...> --job_ids <id1 id2 ...> --directory_path_1 <path> --directory_path_2 <path>"
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
        --job_ids)
            shift
            job_ids=()
            while [[ "$#" -gt 0 && "$1" != --* ]]; do
                job_ids+=("$1")
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
if [[ -z "${experiment_names[*]}" || -z "${job_ids[*]}" || -z "$directory_path_1" || -z "$directory_path_2" ]]; then
    usage
fi

# Check if the two lists are the same size
if [[ ${#experiment_names[@]} -ne ${#job_ids[@]} ]]; then
    echo "The number of experiment names and job IDs must be equal!"
    exit 1
fi

# Iterate over experiment names and job IDs
for i in "${!experiment_names[@]}"; do
    experiment_name="${experiment_names[$i]}"
    job_id="${job_ids[$i]}"
    
    # Define paths to relevant files
    experiment_dir="$directory_path_1/$experiment_name"
    eval_logs="$experiment_dir/eval_log.txt"
    all_evals="$experiment_dir/all_evals.txt"
    slurm_script="$directory_path_2/slurm-$job_id.out"

    # Ensure the experiment directory exists
    if [[ ! -d "$experiment_dir" ]]; then
        echo "Experiment directory $experiment_dir does not exist, skipping..."
        continue
    fi

    # Ensure the eval_logs.txt exists
    if [[ ! -f "$eval_logs" ]]; then
        echo "eval_log.txt not found in $experiment_dir, skipping..."
        continue
    fi

    # Ensure the slurm script file exists
    if [[ ! -f "$slurm_script" ]]; then
        echo "Slurm script $slurm_script not found, skipping..."
        continue
    fi

    # Create (or overwrite) the all_evals.txt file by copying the content of eval_logs.txt
    cp "$eval_logs" "$all_evals"

    # Search for the line "Validity over 10000 molecules" and extract that line and the next 5 lines
    slurm_content=$(awk '/Validity over 10000 molecules/{flag=1;count=6} flag{print; if(--count==0)exit}' "$slurm_script")

    # Append the slurm content to the all_evals.txt file
    echo "$slurm_content" >> "$all_evals"
done

 