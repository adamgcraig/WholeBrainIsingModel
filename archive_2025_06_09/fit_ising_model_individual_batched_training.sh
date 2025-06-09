#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --nodelist=hkbugpusrv06
#SBATCH --mem=5G
#SBATCH --gres=gpu:1
#SBATCH --error=results/errors/fit_ising_model_individual_batched_training_%j.err
#SBATCH --output=results/outs/fit_ising_model_individual_batched_training_%j.out
#SBATCH --job-name="fit_ising_model_individual_batched_training"

echo ${SLURM_JOB_ID}

srun python fit_ising_model_individual.py --data_directory data --output_directory results/ising_model --data_subset training --threshold 0.1 --num_nodes 21 --num_time_points 4800 --reps_per_batch 10 --num_batches 100 --window_length 50 --epochs_per_save 2000 --num_saves 2 --test_length 4800 --compute_fim True
