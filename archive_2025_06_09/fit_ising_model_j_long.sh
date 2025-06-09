#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --nodelist=hkbugpusrv05
#SBATCH --mem=5G
#SBATCH --gres=gpu:1
#SBATCH --error=results/errors/fit_ising_model_j_long_%j.err
#SBATCH --output=results/outs/fit_ising_model_j_long_%j.out
#SBATCH --job-name="fit_ising_model_j_long"

echo ${SLURM_JOB_ID}

srun python fit_ising_model_j.py --data_directory data --output_directory results/ising_model --data_file_name_part group_training_and_individual_all --output_file_name_part long --models_per_subject 5 --sim_length 120000 --num_updates_scaling 1000 --updates_per_save 1000 --num_saves 10 --learning_rate 0.01