#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --nodelist=hkbugpusrv07
#SBATCH --mem=5G
#SBATCH --gres=gpu:1
#SBATCH --error=results/errors/find_feature_to_param_linear_models_0_perm_1M_%j.err
#SBATCH --output=results/outs/find_feature_to_param_linear_models_0_perm_1M_%j.out
#SBATCH --job-name="find_feature_to_param_linear_models_0_perm_1M"

echo ${SLURM_JOB_ID}

srun python find_feature_to_param_linear_models.py --data_directory results/ising_model --output_directory results/ising_model --permutations 1000000 --training_subject_end 837 --model_file_name_part ising_model_light_all_mean_std_0_medium_init_uncentered_reps_5_lr_0.01_steps_1200_pupd_per_bopt_1000_num_opt_1_bopt_steps_66_popt_steps_10000 --mean_state_file_name_part mean_state_all_mean_std_0 --mean_state_product_file_name_part mean_state_product_all_mean_std_0