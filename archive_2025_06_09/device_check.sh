#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --nodelist=hkbugpusrv06
#SBATCH --mem=5G
#SBATCH --gres=gpu:1
#SBATCH --error=results/errors/device_check_%j.err
#SBATCH --output=results/outs/device_check_%j.out
#SBATCH --job-name="device_check"

echo ${SLURM_JOB_ID}

srun python device_check.py