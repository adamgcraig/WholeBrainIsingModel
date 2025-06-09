#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --nodelist=hkbugpusrv05
#SBATCH --mem=5G
#SBATCH --gres=gpu:1
#SBATCH --error=results/errors/beta_optimize_and_fit_halfbrain_individual_training_%j.err
#SBATCH --output=results/outs/beta_optimize_and_fit_halfbrain_individual_training_%j.out
#SBATCH --job-name="beta_optimize_and_fit_halfbrain_individual_training"

echo ${SLURM_JOB_ID}

srun python find_optimal_beta_halfbrain.py --data_directory data --output_directory results/ising_model --model_type individual --data_subset training --num_parallel 30 --sim_length 12000 --num_updates 10
srun python fit_ising_model --data_directory data --output_directory results/ising_model --model_type individual --data_subset training --part halfbrain --num_parallel 30 --sim_length 12000 --num_updates_beta 10 --num_updates 10000 --sims_per_save 100 --learning_rate 0.01