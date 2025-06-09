#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --nodelist=hkbugpusrv08
#SBATCH --mem=5G
#SBATCH --gres=gpu:1
#SBATCH --error=results/errors/print_memory_usage_%j.err
#SBATCH --output=results/outs/print_memory_usage_%j.out
#SBATCH --job-name="print_memory_usage"

echo ${SLURM_JOB_ID}
echo ${SLURMD_NODENAME}
srun python print_memory_usage.py