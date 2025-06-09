#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --nodelist=hkbugpusrv06
#SBATCH --mem=5G
#SBATCH --gres=gpu:1
#SBATCH --time=14-00:00
#SBATCH --error=results/errors/fit_ising_models_pseudolikelihood_multiregion_training_%j.err
#SBATCH --output=results/outs/fit_ising_models_pseudolikelihood_multiregion_training_%j.out
#SBATCH --job-name="fit_ising_models_pseudolikelihood_multiregion_training"

echo ${SLURM_JOB_ID}

srun python fit_ising_models_pseudolikelihood_multiregion.py --data_directory data --output_directory results/ising_model --subjects_start 0 --subjects_end 699 --data_subset training