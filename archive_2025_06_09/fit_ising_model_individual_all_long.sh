#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --nodelist=hkbugpusrv02
#SBATCH --mem=5G
#SBATCH --gres=gpu:1
#SBATCH --error=results/errors/fit_ising_model_group_training_and_individual_all_long_%j.err
#SBATCH --output=results/outs/fit_ising_model_group_training_and_individual_all_long_%j.out
#SBATCH --job-name="fit_ising_model_group_training_and_individual_all_long"

echo ${SLURM_JOB_ID}

srun python fit_ising_model.py --data_directory data --output_directory results/ising_model --data_set group_training_and_individual_all --num_folds 1 --num_betas_per_target 5 --sim_length 12000 --num_beta_saves 10 --beta_updates_per_save 10 --learning_rate 0.01 --num_param_saves 1000000 --param_updates_per_save 100