#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --nodelist=hkbugpusrv06
#SBATCH --mem=5G
#SBATCH --gres=gpu:1
#SBATCH --error=results/errors/find_unique_states_and_counts_mean_std_0_%j.err
#SBATCH --output=results/outs/find_unique_states_and_counts_mean_std_0_%j.out
#SBATCH --job-name="find_unique_states_and_counts_mean_std_0"

echo ${SLURM_JOB_ID}

srun python find_unique_states_and_counts.py --data_directory results/ising_model --output_directory results/ising_model --threshold 0.0