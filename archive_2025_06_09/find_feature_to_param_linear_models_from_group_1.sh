#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --nodelist=hkbugpusrv07
#SBATCH --mem=5G
#SBATCH --gres=gpu:1
#SBATCH --error=results/errors/find_feature_to_param_linear_models_from_group_1_1M_%j.err
#SBATCH --output=results/outs/find_feature_to_param_linear_models_from_group_1_1M_%j.out
#SBATCH --job-name="find_feature_to_param_linear_models_from_group_1_1M"

echo ${SLURM_JOB_ID}

srun python find_feature_to_param_linear_models.py --data_directory results/ising_model --output_directory results/ising_model --permutations 1000000 --training_subject_end 837 --model_file_name_part light_group_threshold_1_betas_5_min_1e-10_max_0.01_beta_steps_1200_param_steps_1200_lr_0.01_beta_updates_68_param_updates_10000_individual_updates_45000 --mean_state_file_name_part mean_state_all_mean_std_1 --mean_state_product_file_name_part mean_state_product_all_mean_std_1