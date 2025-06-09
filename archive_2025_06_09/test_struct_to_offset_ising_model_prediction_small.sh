#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --nodelist=hkbugpusrv05
#SBATCH --mem=5G
#SBATCH --gres=gpu:1
#SBATCH --error=results/errors/test_struct_to_offset_ising_model_prediction_small_%j.err
#SBATCH --output=results/outs/test_struct_to_offset_ising_model_prediction_small_%j.out
#SBATCH --job-name="test_struct_to_offset_ising_model_prediction_small"

echo ${SLURM_JOB_ID}

srun python test_struct_to_offset_ising_model_prediction.py --data_dir data --model_dir results/ising_model --stats_dir results/ising_model --num_nodes 21 --fim_param_string nodes_21_window_50_lr_0.000_threshold_0.100_beta_0.500_reps_1000_epoch_4 --struct_to_offset_param_string depth_0_width_2_batch_10000_lr_0.01 --beta 0.5 --num_steps 48000 --z_score True
