#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --nodelist=hkbugpusrv01
#SBATCH --mem=5G
#SBATCH --gres=gpu:1
#SBATCH --error=results/errors/fit_ising_model_to_group_%j.err
#SBATCH --output=results/outs/fit_ising_model_to_group_%j.out
#SBATCH --job-name="fit_ising_model_to_group"

echo ${SLURM_JOB_ID}

srun python fit_ising_model_to_group.py --data_directory data --output_directory results/ising_model --num_epochs 10000 --epochs_per_save 1 --epochs_per_test 1 --test_length 48000 --num_reps 1000 --num_nodes 21 --window_length 50 --batch_size 1000