#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --nodelist=hkbugpusrv06
#SBATCH --mem=5G
#SBATCH --gres=gpu:1
#SBATCH --error=results/errors/fit_ising_models_window_length_test_%j.err
#SBATCH --output=results/outs/fit_ising_models_window_length_test_%j.out
#SBATCH --job-name="fit_ising_models_window_length_test"

echo ${SLURM_JOB_ID}

srun python fit_ising_models_window_length_test.py --data_directory data --output_directory results/ising_model --max_window_length 4800 --num_epochs 10000 --epochs_per_save 100
