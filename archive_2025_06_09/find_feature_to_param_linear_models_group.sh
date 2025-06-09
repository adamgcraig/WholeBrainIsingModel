#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --nodelist=hkbugpusrv04
#SBATCH --mem=5G
#SBATCH --gres=gpu:1
#SBATCH --error=results/errors/find_feature_to_param_linear_models_group_1M_1M_%j.err
#SBATCH --output=results/outs/find_feature_to_param_linear_models_group_1M_1M_%j.out
#SBATCH --job-name="find_feature_to_param_linear_models_group_1M_1M"

echo ${SLURM_JOB_ID}

srun python find_feature_to_param_linear_models_group.py --data_directory results/ising_model --output_directory results/ising_model --num_region_permutations 1000000 --num_region_pair_permutations 1000000 --model_file_name_part ising_model_light_group_thresholds_31_min_0_max_3_betas_101_min_1e-10_max_0.05_steps_1200_lr_0.01_beta_updates_8_param_updates_40000 --data_file_name_part thresholds_31_min_0_max_3