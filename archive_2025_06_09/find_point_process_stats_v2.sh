#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --nodelist=hkbugpusrv03
#SBATCH --mem=5G
#SBATCH --gres=gpu:1
#SBATCH --error=results/errors/find_point_process_stats_%j.err
#SBATCH --output=results/outs/find_point_process_%j.out
#SBATCH --job-name="find_point_process_short"

echo ${SLURM_JOB_ID}
#     --max_samples_per_batch 1000
srun python find_point_process_stats_from_start.py --input_directory results/ising_model --output_directory results/ising_model --data_subset all --file_name_fragment as_is --num_thresholds 1001 --min_threshold 0 --max_threshold 4 --num_passes 1000 --num_tries_per_pass 1000 --num_permutations 0 --p_value_sample_size 10000