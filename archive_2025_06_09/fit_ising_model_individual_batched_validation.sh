#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --nodelist=hkbugpusrv04
#SBATCH --mem=5G
#SBATCH --gres=gpu:1
#SBATCH --error=results/errors/fit_ising_model_individual_batched_validation_%j.err
#SBATCH --output=results/outs/fit_ising_model_individual_batched_validation_%j.out
#SBATCH --job-name="fit_ising_model_individual_batched_validation"

echo ${SLURM_JOB_ID}

srun python fit_ising_model_individual.py --data_directory data --output_directory results/ising_model --data_subset validation --threshold 0.1 --num_nodes 21 --num_time_points 4800 --reps_per_batch 100 --num_batches 10 --window_length 50 --epochs_per_save 2000 --num_saves 2 --test_length 4800
