#!/bin/bash
#SBATCH --nodes=8
#SBATCH --ntasks=8
#SBATCH --nodelist=hkbugpusrv01,hkbugpusrv02,hkbugpusrv03,hkbugpusrv04,hkbugpusrv05,hkbugpusrv06,hkbugpusrv07,hkbugpusrv08
#SBATCH --mem=5G
#SBATCH --gres=gpu:1
#SBATCH --error=results/errors/gpustat_%j.err
#SBATCH --output=results/outs/gpustat_%j.out
#SBATCH --job-name="gpustat"

echo ${SLURM_JOB_ID}
echo ${SLURMD_NODENAME}
srun gpustat --no-color