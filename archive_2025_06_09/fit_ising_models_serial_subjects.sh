#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --nodelist=hkbugpusrv01
#SBATCH --mem=5G
#SBATCH --gres=gpu:1
#SBATCH --error=results/errors/fit_ising_models_serial_subjects_21_nodes_%j.err
#SBATCH --output=results/outs/fit_ising_models_serial_subjects_21_nodes_%j.out
#SBATCH --job-name="fit_ising_models_serial_subjects_21_nodes"

echo ${SLURM_JOB_ID}

srun python fit_ising_models_serial_subjects.py --data_directory data --output_directory results/ising_model --num_nodes 21 --num_reps 100 --windows_per_epoch 96 --epochs_per_print 10 --prints_per_save 10 --num_saves 3
