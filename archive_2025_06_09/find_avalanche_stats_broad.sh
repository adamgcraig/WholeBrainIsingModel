#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --nodelist=hkbugpusrv06
#SBATCH --mem=5G
#SBATCH --gres=gpu:1
#SBATCH --error=results/errors/find_avalanche_stats_broad_%j.err
#SBATCH --output=results/outs/find_avalanche_stats_broad_%j.out
#SBATCH --job-name="find_avalanche_stats_broad"

echo ${SLURM_JOB_ID}

srun python find_avalanche_stats.py --input_directory results/ising_model --output_directory results/ising_model --data_subset all --file_name_fragment as_is --num_thresholds 701 --max_threshold 6