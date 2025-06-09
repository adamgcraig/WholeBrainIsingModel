#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --nodelist=hkbugpusrv02
#SBATCH --mem=5G
#SBATCH --gres=gpu:1
#SBATCH --error=results/errors/compute_optimal_beta_for_group_%j.err
#SBATCH --output=results/outs/compute_optimal_beta_for_group_%j.out
#SBATCH --job-name="compute_optimal_beta_for_group"

echo ${SLURM_JOB_ID}

srun python compute_optimal_beta_for_group.py --data_directory data --output_directory results/ising_model --data_subset training --num_parallel 10000 --sim_length 120000 --num_updates 3