#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --nodelist=hkbugpusrv03
#SBATCH --mem=5G
#SBATCH --gres=gpu:1
#SBATCH --error=results/errors/find_feature_to_param_mlp_models_cross_validation_from_group_1_10_10_1_batch_%j.err
#SBATCH --output=results/outs/find_feature_to_param_mlp_models_cross_validation_from_group_1_10_10_1_batch_%j.out
#SBATCH --job-name="find_feature_to_param_mlp_models_cross_validation_from_group_1_10_10_1_batch"

echo ${SLURM_JOB_ID}

srun python find_feature_to_param_mlp_models_cross_validation.py --data_directory results/ising_model --output_directory results/ising_model --num_training_subjects 670 --num_region_permutations 10 --num_region_pair_permutations 10 --model_file_name_part ising_model_light_group_threshold_1_betas_5_min_1e-10_max_0.01_beta_steps_1200_param_steps_1200_lr_0.01_beta_updates_68_param_updates_10000_individual_updates_45000 --data_file_name_part all_mean_std_1 --save_all --learning_rate 0.0001 --num_node_hidden_layers 3 --node_hidden_layer_width 4 --num_edge_hidden_layers 3 --edge_hidden_layer_width 4 --max_epochs 1000000 --patience 1000 --min_improvement 0.001 --node_batch_size 670 --edge_batch_size 670