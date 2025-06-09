#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --nodelist=hkbugpusrv04
#SBATCH --mem=5G
#SBATCH --gres=gpu:1
#SBATCH --error=results/errors/find_struct_to_ising_fisher_information_21_nodes_training_%j.err
#SBATCH --output=results/outs/find_struct_to_ising_fisher_information_21_nodes_training_%j.out
#SBATCH --job-name="find_struct_to_ising_fisher_information_21_nodes_training"

echo ${SLURM_JOB_ID}

srun python find_struct_to_ising_fisher_information.py --data_dir data --model_dir results/ising_model --stats_dir results/ising_model --data_subset training --num_nodes 21 --num_steps 48000 --model_param_string struct2ising_epochs_500_val_batch_100_steps_4800_lr_0.0001_batches_1000_node_hl_2_node_w_21_edge_hl_2_edge_w_441_ising_nodes_21_reps_100_epochs_1000_window_50_lr_0.001_threshold_0.100
