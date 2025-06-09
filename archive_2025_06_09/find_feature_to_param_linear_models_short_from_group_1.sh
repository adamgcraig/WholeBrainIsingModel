#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --nodelist=hkbugpusrv05
#SBATCH --mem=5G
#SBATCH --gres=gpu:1
#SBATCH --error=results/errors/find_feature_to_param_linear_models_short_from_group_1_1M_20k_%j.err
#SBATCH --output=results/outs/find_feature_to_param_linear_models_short_from_group_1_1M_20k_%j.out
#SBATCH --job-name="find_feature_to_param_linear_models_short_from_group_1_1M_20k"

echo ${SLURM_JOB_ID}

srun python find_feature_to_param_linear_models_short.py --data_directory results/ising_model --output_directory results/ising_model --num_region_permutations 1000000 --num_region_pair_permutations 20000 --model_file_name_part ising_model_light_group_threshold_1_betas_5_min_1e-10_max_0.01_beta_steps_1200_param_steps_1200_lr_0.01_beta_updates_68_param_updates_10000_individual_updates_45000 --data_file_name_part all_mean_std_1