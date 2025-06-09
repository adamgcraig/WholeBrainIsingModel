#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --nodelist=hkbugpusrv01
#SBATCH --mem=5G
#SBATCH --gres=gpu:1
#SBATCH --error=results/errors/find_param_pca_%j.err
#SBATCH --output=results/outs/find_param_pca_%j.out
#SBATCH --job-name="find_param_pca"

echo ${SLURM_JOB_ID}

srun python find_param_pca.py --data_directory results/ising_model --output_directory results/ising_model --iteration_increment 10000 --scale_by_beta