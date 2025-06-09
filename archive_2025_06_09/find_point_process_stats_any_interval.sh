#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --nodelist=hkbugpusrv03
#SBATCH --mem=5G
#SBATCH --gres=gpu:1
#SBATCH --error=results/errors/find_point_process_stats_any_interval_%j.err
#SBATCH --output=results/outs/find_point_process_stats_any_interval_%j.out
#SBATCH --job-name="find_point_process_stats_any_interval"

echo ${SLURM_JOB_ID}

srun python find_point_process_stats_any_interval.py --input_directory results/ising_model --output_directory results/ising_model --num_thresholds 5 --min_threshold 0 --max_threshold 4 --initial_min_exponent 1 --initial_max_exponent 10 --num_passes 10 --num_tries_per_pass 1000 --precision 2e-6 --num_sample_ks_distances 100