#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --nodelist=hkbugpusrv07
#SBATCH --mem=5G
#SBATCH --gres=gpu:1
#SBATCH --error=results/errors/fit_ising_model_light_medium_mean_std_0.5_flip_rate_%j.err
#SBATCH --output=results/outs/fit_ising_model_light_medium_mean_std_0.5_flip_rate_%j.out
#SBATCH --job-name="fit_ising_model_light_medium_flip_rate_mean_std_0.5"

echo ${SLURM_JOB_ID}

srun python fit_ising_model_light.py --data_directory results/ising_model --output_directory results/ising_model --data_file_name_part all_mean_std_0.5 --output_file_name_part medium --models_per_subject 5 --sim_length 1200 --num_updates_beta 10000 --updates_per_beta_reopt 100 --beta_reopts_per_save 10 --num_saves 10 --learning_rate 0.01 --combine_scans --target_flip_rate 0.34 --max_beta 100