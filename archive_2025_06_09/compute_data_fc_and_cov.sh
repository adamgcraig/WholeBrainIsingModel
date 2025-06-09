#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --nodelist=hkbugpusrv04
#SBATCH --mem=5G
#SBATCH --gres=gpu:1
#SBATCH --error=results/errors/compute_data_fc_and_cov_%j.err
#SBATCH --output=results/outs/compute_data_fc_and_cov_%j.out
#SBATCH --job-name="compute_data_fc_and_cov"

echo ${SLURM_JOB_ID}

srun python compute_data_fc_and_cov.py --data_directory data --output_directory results/ising_model --data_subset training --threshold 0.1 --num_nodes 21 --window_length 3211200