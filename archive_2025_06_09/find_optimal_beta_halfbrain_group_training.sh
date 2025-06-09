#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --nodelist=hkbugpusrv06
#SBATCH --mem=5G
#SBATCH --gres=gpu:1
#SBATCH --error=results/errors/find_optimal_beta_halfbrain_group_training_%j.err
#SBATCH --output=results/outs/find_optimal_beta_halfbrain_group_training_%j.out
#SBATCH --job-name="find_optimal_beta_halfbrain_group_training"

echo ${SLURM_JOB_ID}

srun python find_optimal_beta_halfbrain.py --data_directory data --output_directory results/ising_model --model_type group --data_subset training --num_parallel 10000 --sim_length 48000 --num_updates 10