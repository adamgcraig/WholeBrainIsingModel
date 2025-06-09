#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --nodelist=hkbugpusrv01
#SBATCH --mem=5G
#SBATCH --gres=gpu:1
#SBATCH --error=results/errors/fit_ising_models_every_step_360_nodes_validation_%j.err
#SBATCH --output=results/outs/fit_ising_models_every_step_360_nodes_validation_%j.out
#SBATCH --job-name="fit_ising_models_every_step_360_nodes_validation"

echo ${SLURM_JOB_ID}

srun python fit_ising_models_every_step.py --data_directory data --output_directory results/ising_model --num_reps 100 --subjects_start 0 --subjects_end 83 --num_epochs 200 --num_nodes 360 --data_subset validation
