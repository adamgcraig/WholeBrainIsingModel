#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --nodelist=hkbugpusrv05
#SBATCH --mem=5G
#SBATCH --gres=gpu:1
#SBATCH --error=results/errors/fit_ising_models_parallel_sims_21_nodes_%j.err
#SBATCH --output=results/outs/fit_ising_models_parallel_sims_21_nodes_%j.out
#SBATCH --job-name="fit_ising_models_parallel_sims_21_nodes"

echo ${SLURM_JOB_ID}

srun python fit_ising_model_parallel_sims.py --data_directory data --output_directory results/ising_model --num_epochs 1000 --epochs_per_save 100 --num_sims 2400 --num_nodes 21 --window_length 50
