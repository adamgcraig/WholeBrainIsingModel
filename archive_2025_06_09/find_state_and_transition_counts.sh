#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --nodelist=hkbugpusrv07
#SBATCH --mem=5G
#SBATCH --gres=gpu:1
#SBATCH --error=results/errors/find_state_and_transition_counts_%j.err
#SBATCH --output=results/outs/find_state_and_transition_counts_%j.out
#SBATCH --job-name="find_state_and_transition_counts"

echo ${SLURM_JOB_ID}

srun python find_state_and_transition_counts.py --data_directory data --output_directory results/ising_model --data_subset training