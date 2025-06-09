#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --nodelist=hkbugpusrv03
#SBATCH --mem=5G
#SBATCH --gres=gpu:1
#SBATCH --error=results/errors/prep_structural_features_%j.err
#SBATCH --output=results/outs/prep_structural_features_%j.out
#SBATCH --job-name="prep_structural_features"

echo ${SLURM_JOB_ID}

srun python prep_structural_features.py --data_directory data --output_directory results/ising_model