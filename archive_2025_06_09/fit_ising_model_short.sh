#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --nodelist=hkbugpusrv06
#SBATCH --mem=5G
#SBATCH --gres=gpu:1
#SBATCH --error=results/errors/fit_ising_model_short_%j.err
#SBATCH --output=results/outs/fit_ising_model_short_%j.out
#SBATCH --job-name="fit_ising_model_short"

echo ${SLURM_JOB_ID}

srun python fit_ising_model_short.py --data_directory results/ising_model --model_directory results/ising_model --mean_state_file_name data_mean_state_aal_short_threshold_1.pt --mean_state_product_file_name data_mean_state_product_aal_short_threshold_1.pt --model_file_name_prefix ising_model_aal_short_threshold_1 --models_per_subject 101 --beta_sim_length 1200 --param_sim_length 1200 --max_num_beta_updates 1000000 --updates_per_save 1000 --num_saves 1000 --learning_rate 0.01 --max_beta 1.0