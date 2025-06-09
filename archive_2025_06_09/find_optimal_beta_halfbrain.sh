#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --nodelist=hkbugpusrv03
#SBATCH --mem=5G
#SBATCH --gres=gpu:1
#SBATCH --error=results/errors/find_optimal_beta_halfbrain_group_training_%j.err
#SBATCH --output=results/outs/find_optimal_beta_halfbrain_group_training_%j.out
#SBATCH --job-name="find_optimal_beta_halfbrain_group_training"

echo ${SLURM_JOB_ID}

srun python find_optimal_beta_halfbrain.py --data_directory data --output_directory results/ising_model --data_subset validation --num_parallel 25 --sim_length 120000 --num_updates 8