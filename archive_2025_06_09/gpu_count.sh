#!/bin/bash
#SBATCH --nodes=8
#SBATCH --ntasks=8
#SBATCH --nodelist=hkbugpusrv08
#SBATCH --mem=5G
#SBATCH --gres=gpu:1
#SBATCH --error=results/errors/gpu_count_%j.err
#SBATCH --output=results/outs/gpu_count_%j.out
#SBATCH --job-name="gpu_count"

echo ${SLURM_JOB_ID}

srun python gpu_count.py