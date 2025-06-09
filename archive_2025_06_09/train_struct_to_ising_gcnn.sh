#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --nodelist=hkbugpusrv02
#SBATCH --mem=5G
#SBATCH --gres=gpu:1
#SBATCH --error=results/errors/train_struct_to_ising_gcnn_%j.err
#SBATCH --output=results/outs/train_struct_to_ising_gcnn_%j.out
#SBATCH --job-name="train_struct_to_ising_gcnn"

echo ${SLURM_JOB_ID}

srun python train_struct_to_ising_gcnn.py --data_directory data --output_directory results/ising_model --node_features_file node_features_group_training_and_individual_all.pt --edge_features_file edge_features_group_training_and_individual_all.pt --ising_model_file ising_model_group_training_and_individual_all_fold_1_betas_5_steps_1200_lr_0.01.pt --training_start 1 --training_end 670 --validation_start 670 --validation_end 754 --num_graph_convolution_layers 3 --num_mlp_hidden_layers 3 --rep_dims 7 --num_saves 1000000 --num_epochs_per_save 1000 --batch_size 100 --learning_rate 0.000001