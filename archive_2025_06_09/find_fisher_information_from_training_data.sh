#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --nodelist=hkbugpusrv01
#SBATCH --mem=5G
#SBATCH --gres=gpu:1
#SBATCH --error=results/errors/find_fisher_information_from_training_data_%j.err
#SBATCH --output=results/outs/find_fisher_information_from_training_data_%j.out
#SBATCH --job-name="find_fisher_information_from_training_data"

echo ${SLURM_JOB_ID}

srun python find_fisher_information_from_training_data.py --data_dir data --output_dir results/ising_model
