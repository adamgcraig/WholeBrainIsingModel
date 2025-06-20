#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --nodelist=hkbugpusrv04
#SBATCH --mem=5G
#SBATCH --gres=gpu:1
#SBATCH --error=results/errors/find_mean_state_and_state_product_feature_correlations_for_threshold_%j.err
#SBATCH --output=results/outs/find_mean_state_and_state_product_feature_correlations_for_threshold_%j.out
#SBATCH --job-name="find_mean_state_and_state_product_feature_correlations_for_threshold"

echo ${SLURM_JOB_ID}

srun python find_mean_state_and_state_product_feature_correlations_and_p_values_for_threshold.py --input_directory results/ising_model --output_directory results/ising_model