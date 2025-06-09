#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --nodelist=hkbugpusrv06
#SBATCH --mem=5G
#SBATCH --gres=gpu:1
#SBATCH --error=results/errors/find_struct_to_offset_fisher_information_small_%j.err
#SBATCH --output=results/outs/find_struct_to_offset_fisher_information_small_%j.out
#SBATCH --job-name="find_struct_to_offset_fisher_information_small"

echo ${SLURM_JOB_ID}

srun python find_struct_to_offset_fisher_information.py --data_dir data --model_dir results/ising_model --stats_dir results/ising_model --num_nodes 21 --fim_param_string nodes_21_window_50_lr_0.000_threshold_0.100_beta_0.500_reps_1000_epoch_4 --struct_to_offset_param_string depth_0_width_2_batch_10000_lr_0.01 --num_reps_group 1000 --epoch_group 4 --beta 0.5 --num_steps 48000 --z_score True --num_optimizer_steps 200000 --optimizer_learning_rate 0.00001 --optimizer_print_every_steps 100
