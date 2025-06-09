#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --nodelist=hkbugpusrv07
#SBATCH --mem=5G
#SBATCH --gres=gpu:1
#SBATCH --error=results/errors/divide_g2gcnn_examples_rectangular_max_rmse_2_%j.err
#SBATCH --output=results/outs/divide_g2gcnn_examples_rectangular_max_rmse_2_%j.out
#SBATCH --job-name="divide_g2gcnn_examples_rectangular_max_rmse_2"

echo ${SLURM_JOB_ID}

srun python divide_g2gcnn_examples.py --input_directory results/ising_model --output_directory results/g2gcnn_examples --node_features_file node_features_group_training_and_individual_all_rectangular.pt --edge_features_file edge_features_group_training_and_individual_all.pt --ising_model_file ising_model_beta_updates_100_param_updates_3100_group_training_and_individual_all_fold_1_betas_5_steps_1200_lr_0.01.pt --ising_model_rmse_file combined_mean_state_rmse_sim_length_120000_beta_updates_100_param_updates_3100_group_training_and_individual_all_fold_1_betas_5_steps_1200_lr_0.01.pt --coordinate_type_name rectangular --max_ising_model_rmse 2.0 --training_start 1 --training_end 670 --validation_start 670 --validation_end 754