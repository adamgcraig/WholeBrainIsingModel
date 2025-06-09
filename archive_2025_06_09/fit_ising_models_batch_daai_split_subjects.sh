#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --nodelist=hkbugpusrv01
#SBATCH --mem=5G
#SBATCH --gres=gpu:1
#SBATCH --time=14-00:00
#SBATCH --error=results/errors/fit_ising_models_batch_daai_split_subjects_0_%j.err
#SBATCH --output=results/outs/fit_ising_models_batch_daai_split_subjects_0_%j.out
#SBATCH --job-name="fit_ising_models_batch_daai_split_subjects_0"

echo ${SLURM_JOB_ID}

srun python fit_ising_models_batch_daai_split_subjects.py --num_nodes 360 --num_steps 50 --num_epochs 200 --subjects_start 0 --subjects_end 67