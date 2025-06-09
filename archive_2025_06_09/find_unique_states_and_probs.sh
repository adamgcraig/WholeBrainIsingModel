#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --nodelist=hkbugpusrv07
#SBATCH --mem=5G
#SBATCH --gres=gpu:1
#SBATCH --error=results/errors/find_unique_states_and_probs_training_360_thresh_median_%j.err
#SBATCH --output=results/outs/find_unique_states_and_probs_training_360_thresh_median_%j.out
#SBATCH --job-name="find_unique_states_and_probs_training_360_thresh_median"

echo ${SLURM_JOB_ID}

srun python find_unique_states_and_probs.py --data_directory data --output_directory results/ising_model --data_subset training --threshold median