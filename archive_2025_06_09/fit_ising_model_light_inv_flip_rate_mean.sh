#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --nodelist=hkbugpusrv04
#SBATCH --mem=5G
#SBATCH --gres=gpu:1
#SBATCH --error=results/errors/fit_ising_model_light_inv_mean_flip_rate_%j.err
#SBATCH --output=results/outs/fit_ising_model_light_inv_mean_flip_rate_%j.out
#SBATCH --job-name="fit_ising_model_light_inv_flip_rate_mean"

echo ${SLURM_JOB_ID}

srun python fit_ising_model_light.py --data_directory results/ising_model --output_directory results/ising_model --data_file_name_part all_mean_std_0 --output_file_name_part inv --models_per_subject 5 --sim_length 1200 --num_updates_beta 1000000 --updates_per_save 1000 --saves_per_beta_opt 1000 --num_beta_opts 1 --learning_rate 0.01 --combine_scans --target_flip_rate 0.34 --max_beta 1 --use_inverse_cov --center_cov