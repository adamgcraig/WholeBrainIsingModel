#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --nodelist=hkbugpusrv01
#SBATCH --mem=5G
#SBATCH --gres=gpu:1
#SBATCH --error=results/errors/find_feature_to_param_linear_models_cross_validation_from_group_0_10k_10k_%j.err
#SBATCH --output=results/outs/find_feature_to_param_linear_models_cross_validation_from_group_0_10k_10k_%j.out
#SBATCH --job-name="find_feature_to_param_linear_models_cross_validation_from_group_0_10k_10k"

echo ${SLURM_JOB_ID}

srun python find_feature_to_param_linear_models_cross_validation.py --data_directory results/ising_model --output_directory results/ising_model --num_training_subjects 670 --num_region_permutations 10000 --num_region_pair_permutations 10000 --model_file_name_part ising_model_light_group_threshold_0_betas_5_min_1e-10_max_0.01_beta_steps_1200_param_steps_1200_lr_0.01_beta_updates_70_param_updates_10000_individual_updates_10000 --data_file_name_part all_mean_std_0 --save_all