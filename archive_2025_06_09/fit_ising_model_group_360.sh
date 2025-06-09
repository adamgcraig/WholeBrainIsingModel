#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --nodelist=hkbugpusrv02
#SBATCH --mem=5G
#SBATCH --gres=gpu:1
#SBATCH --error=results/errors/fit_ising_model_group_training_360_thresh_pt1_%j.err
#SBATCH --output=results/outs/fit_ising_model_group_training_360_thresh_pt1_%j.out
#SBATCH --job-name="fit_ising_model_group_training_360_thresh_pt1"

echo ${SLURM_JOB_ID}

srun python fit_ising_model_group.py --data_directory data --output_directory results/ising_model --data_subset training --threshold 0.1 --num_nodes 360 --num_time_points 4800 --reps_per_batch 10 --num_batches 1 --window_length 139 --epochs_per_save 1 --num_saves 1000 --test_length 48000 --compute_fim False --cube_update False