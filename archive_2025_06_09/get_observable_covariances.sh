#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --nodelist=hkbugpusrv08
#SBATCH --mem=5G
#SBATCH --gres=gpu:1
#SBATCH --error=results/errors/get_observable_covariances_%j.err
#SBATCH --output=results/outs/get_observable_covariances_%j.out
#SBATCH --job-name="get_observable_covariances"

echo ${SLURM_JOB_ID}

srun python get_observable_covariances.py --input_directory results/ising_model --output_directory results/ising_model --threshold 1.0