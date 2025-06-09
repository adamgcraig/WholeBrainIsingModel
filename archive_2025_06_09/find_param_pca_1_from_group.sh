#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --nodelist=hkbugpusrv08
#SBATCH --mem=5G
#SBATCH --gres=gpu:1
#SBATCH --error=results/errors/find_param_pca_1_from_group_%j.err
#SBATCH --output=results/outs/find_param_pca_1_from_group_%j.out
#SBATCH --job-name="find_param_pca_1_from_group"

echo ${SLURM_JOB_ID}

srun python find_param_pca.py --data_directory results/ising_model --output_directory results/ising_model --model_file_name_part ising_model_light_group_threshold_1_betas_5_min_1e-10_max_0.01_beta_steps_1200_param_steps_1200_lr_0.01_beta_updates_68_param_updates_10000_individual_updates_45000 --mean_state_file_name_part mean_state_all_mean_std_1 --mean_state_product_file_name_part mean_state_product_all_mean_std_1 --scale_by_beta