#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --nodelist=hkbugpusrv02
#SBATCH --mem=5G
#SBATCH --gres=gpu:1
#SBATCH --error=results/errors/make_inverse_cov_group_multithreshold_check_%j.err
#SBATCH --output=results/outs/make_inverse_cov_group_multithreshold_check_%j.out
#SBATCH --job-name="make_inverse_cov_group_multithreshold_check"

echo ${SLURM_JOB_ID}

srun python make_inverse_cov_group_multithreshold_check.py --input_directory results/ising_model --output_directory results/ising_model --beta_opt_length 12000