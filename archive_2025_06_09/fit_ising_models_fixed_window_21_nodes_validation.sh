#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --nodelist=hkbugpusrv03
#SBATCH --mem=5G
#SBATCH --gres=gpu:1
#SBATCH --time=14-00:00
#SBATCH --error=results/errors/fit_ising_models_fixed_window_21_nodes_validation_%j.err
#SBATCH --output=results/outs/fit_ising_models_fixed_window_21_nodes_validation_%j.out
#SBATCH --job-name="fit_ising_models_fixed_window_21_nodes_validation"

echo ${SLURM_JOB_ID}

srun python fit_ising_models_fixed_window.py --data_directory data --output_directory results/ising_model --window_length 50 --num_reps 10 --subjects_start 0 --subjects_end 83 --num_epochs 200 --num_nodes 21 --data_subset validation
