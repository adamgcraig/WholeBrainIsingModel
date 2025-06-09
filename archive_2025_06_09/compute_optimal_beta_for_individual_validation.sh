#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --nodelist=hkbugpusrv03
#SBATCH --mem=5G
#SBATCH --gres=gpu:1
#SBATCH --error=results/errors/compute_optimal_beta_for_individual_validation_%j.err
#SBATCH --output=results/outs/compute_optimal_beta_for_individual_validation_%j.out
#SBATCH --job-name="compute_optimal_beta_for_individual_validation"

echo ${SLURM_JOB_ID}

srun python compute_optimal_beta_for_individual.py --data_directory data --output_directory results/ising_model --data_subset validation --num_parallel 25 --sim_length 120000 --num_updates 8