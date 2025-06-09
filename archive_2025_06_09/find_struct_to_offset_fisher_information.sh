#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --nodelist=hkbugpusrv07
#SBATCH --mem=5G
#SBATCH --gres=gpu:1
#SBATCH --error=results/errors/find_struct_to_offset_fisher_information_%j.err
#SBATCH --output=results/outs/find_struct_to_offset_fisher_information_%j.out
#SBATCH --job-name="find_struct_to_offset_fisher_information"

echo ${SLURM_JOB_ID}

srun python find_struct_to_offset_fisher_information.py --data_dir data --model_dir results/ising_model --stats_dir results/ising_model --num_nodes 21 --ising_model_param_string nodes_21_window_50_lr_0.000_threshold_0.100_beta_0.500 --struct_to_offset_param_string reps_1000_epoch_4_depth_0_width_2_batch_1000_lr_0.0001 --num_reps_group 1000 --epoch_group 4 --beta 0.5 --num_steps 48000 --z_score True
