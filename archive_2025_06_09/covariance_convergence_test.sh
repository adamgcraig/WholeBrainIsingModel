#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --nodelist=hkbugpusrv01
#SBATCH --mem=5G
#SBATCH --gres=gpu:1
#SBATCH --error=results/errors/covariance_convergence_test_%j.err
#SBATCH --output=results/outs/covariance_convergence_test_%j.out
#SBATCH --job-name="covariance_convergence_test"

echo ${SLURM_JOB_ID}

srun python covariance_convergence_test_v2.py --data_directory data --output_directory results/ising_model --threshold 0.1
