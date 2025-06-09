#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --nodelist=hkbugpusrv04
#SBATCH --mem=5G
#SBATCH --gres=gpu:1
#SBATCH --error=results/errors/cluster_binary_time_series_%j.err
#SBATCH --output=results/outs/cluster_binary_time_series_%j.out
#SBATCH --job-name="cluster_binary_time_series"

echo ${SLURM_JOB_ID}

srun python cluster_binary_time_series.py --data_directory data --output_directory results/ising_model --target_num_nodes 21