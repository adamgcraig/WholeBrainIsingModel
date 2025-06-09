#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --nodelist=hkbugpusrv03
#SBATCH --mem=5G
#SBATCH --gres=gpu:1
#SBATCH --error=results/errors/filter_subject_lists_%j.err
#SBATCH --output=results/outs/filter_subject_lists_%j.out
#SBATCH --job-name="filter_subject_lists"

echo ${SLURM_JOB_ID}

srun python filter_subject_lists.py --input_directory data --output_directory data/ml_examples