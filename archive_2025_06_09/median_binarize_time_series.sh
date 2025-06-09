#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --nodelist=hkbugpusrv07
#SBATCH --mem=5G
#SBATCH --gres=gpu:1
#SBATCH --error=results/errors/median_binarize_time_series_%j.err
#SBATCH --output=results/outs/median_binarize_time_series_%j.out
#SBATCH --job-name="median_binarize_time_series"

echo ${SLURM_JOB_ID}

srun python median_binarize_time_series.py --data_directory data --output_directory results/ising_model --data_subset training