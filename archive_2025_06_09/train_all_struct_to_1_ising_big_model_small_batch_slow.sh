#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --nodelist=hkbugpusrv07
#SBATCH --mem=5G
#SBATCH --gres=gpu:1
#SBATCH --error=results/errors/train_all_struct_to_1_ising_big_model_small_batch_slow_%j.err
#SBATCH --output=results/outs/train_all_struct_to_1_ising_big_model_small_batch_slow_%j.out
#SBATCH --job-name="train_all_struct_to_1_ising_big_model_small_batch_slow"

echo ${SLURM_JOB_ID}

srun python train_all_struct_to_1_ising.py --structural_data_dir data --ising_model_dir results/ising_model --stats_dir results/ising_model --fim_param_string nodes_21_window_50_lr_0.000_threshold_0.100_beta_0.500_reps_1000_epoch_4 --num_nodes 21 --training_batch_size 100 --validation_batch_size 10000 --learning_rate 0.000001 --num_epochs 2500 --num_hidden_layers 10 --hidden_layer_width 1000
