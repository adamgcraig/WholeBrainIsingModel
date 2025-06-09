#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --nodelist=hkbugpusrv03
#SBATCH --mem=5G
#SBATCH --gres=gpu:1
#SBATCH --error=results/errors/covariance_convergence_test_none_%j.err
#SBATCH --output=results/outs/covariance_convergence_test_none_%j.out
#SBATCH --job-name="covariance_convergence_test_none"

echo ${SLURM_JOB_ID}

srun python covariance_convergence_test_v2.py --data_directory data --output_directory results/ising_model --threshold none
