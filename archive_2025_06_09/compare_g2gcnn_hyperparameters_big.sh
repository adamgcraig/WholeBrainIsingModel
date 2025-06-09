#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --nodelist=hkbugpusrv08
#SBATCH --mem=5G
#SBATCH --gres=gpu:1
#SBATCH --error=results/errors/compare_g2gcnn_hyperparameters_big_model_rectangular_max_rmse_2_%j.err
#SBATCH --output=results/outs/compare_g2gcnn_hyperparameters_big_model_rectangular_max_rmse_2_%j.out
#SBATCH --job-name="compare_g2gcnn_hyperparameters_big_model_rectangular_max_rmse_2"

echo ${SLURM_JOB_ID}

srun python compare_g2gcnn_hyperparameters.py --file_directory results/ising_model --ising_model_file ising_model_beta_updates_100_param_updates_3100_group_training_and_individual_all_fold_1_betas_5_steps_1200_lr_0.01.pt --ising_model_rmse_file combined_mean_state_rmse_sim_length_120000_beta_updates_100_param_updates_3100_group_training_and_individual_all_fold_1_betas_5_steps_1200_lr_0.01.pt --results_file_name compare_g2gcnn_hyperparameters_big_rectangular_max_rmse_2.pkl --training_start 1 --training_end 670 --validation_start 670 --validation_end 754 --max_ising_model_rmse 2.0 --num_epochs 3000 --num_graph_convolution_layers 3 --min_graph_convolution_layers 5 --max_graph_convolution_layers 15 --num_mlp_hidden_layers 3 --min_mlp_hidden_layers 5 --max_mlp_hidden_layers 15 --num_rep_dims 3 --min_rep_dims 20 --max_rep_dims 100 --num_batch_sizes 3 --min_batch_size 10 --max_batch_size 100 --num_learning_rates 3 --min_learning_rate 0.001 --max_learning_rate 0.1