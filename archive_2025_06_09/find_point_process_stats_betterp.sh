#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --nodelist=hkbugpusrv07
#SBATCH --mem=5G
#SBATCH --gres=gpu:1
#SBATCH --error=results/errors/find_point_process_stats_betterp_%j.err
#SBATCH --output=results/outs/find_point_process_betterp_%j.out
#SBATCH --job-name="find_point_process_stats_betterp"

echo ${SLURM_JOB_ID}

srun python find_point_process_stats.py --input_directory results/ising_model --output_directory results/ising_model --data_subset all --file_name_fragment as_is --num_thresholds 100 --min_threshold 0 --max_threshold 4 --num_passes 10 --num_tries_per_pass 10 --max_x_min 100 --num_permutations 10000
