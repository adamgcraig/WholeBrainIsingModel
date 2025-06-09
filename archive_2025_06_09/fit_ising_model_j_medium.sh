#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --nodelist=hkbugpusrv04
#SBATCH --mem=5G
#SBATCH --gres=gpu:1
#SBATCH --error=results/errors/fit_ising_model_j_medium_%j.err
#SBATCH --output=results/outs/fit_ising_model_j_medium_%j.out
#SBATCH --job-name="fit_ising_model_j_medium"

echo ${SLURM_JOB_ID}

srun python fit_ising_model_j.py --data_directory data --output_directory results/ising_model --data_file_name_part group_training_and_individual_all --output_file_name_part medium --models_per_subject 5 --sim_length 12000 --num_updates_scaling 1000 --updates_per_save 10 --num_saves 10 --learning_rate 0.001