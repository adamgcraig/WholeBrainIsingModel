#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --nodelist=hkbugpusrv07
#SBATCH --mem=5G
#SBATCH --gres=gpu:1
#SBATCH --error=results/errors/optimize_all_struct_to_all_param_hyperparameters_from_group_1_%j.err
#SBATCH --output=results/outs/optimize_all_struct_to_all_param_hyperparameters_from_group_1_%j.out
#SBATCH --job-name="optimize_all_struct_to_all_param_hyperparameters_from_group_1"

echo ${SLURM_JOB_ID}

srun python optimize_all_struct_to_all_param_hyperparameters.py --data_directory results/ising_model --output_directory results/ising_model --model_file_name_part ising_model_light_group_threshold_1_betas_5_min_1e-10_max_0.01_beta_steps_1200_param_steps_1200_lr_0.01_beta_updates_68_param_updates_10000_individual_updates_45000 --data_file_name_part all_mean_std_1  --num_permutations 10 --max_num_hidden_layers 9 --max_hidden_layer_width 10 --batch_size_increment 67 --max_learning_rate_power 0.0 --max_learning_rate_power 5.0 --max_epochs 1000000 --epochs_per_validation 1000 --min_improvement 0.001 --output_file_name_part all_all_from_group_glasser_1