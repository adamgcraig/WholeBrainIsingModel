#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --nodelist=hkbugpusrv08
#SBATCH --mem=5G
#SBATCH --gres=gpu:1
#SBATCH --error=results/errors/find_feature_to_param_linear_models_%j.err
#SBATCH --output=results/outs/find_feature_to_param_linear_models_%j.out
#SBATCH --job-name="find_feature_to_param_linear_models"

echo ${SLURM_JOB_ID}

srun python find_feature_to_param_linear_models.py --data_directory results/ising_model --output_directory results/ising_model --permutations 1000000 --training_subject_end 837