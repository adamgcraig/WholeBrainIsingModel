#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --nodelist=hkbugpusrv07
#SBATCH --mem=5G
#SBATCH --gres=gpu:1
#SBATCH --error=results/errors/find_mean_state_and_state_product_for_threshold_mean_std_pt5_%j.err
#SBATCH --output=results/outs/find_mean_state_and_state_product_for_threshold_mean_std_pt5_%j.out
#SBATCH --job-name="find_mean_state_and_state_product_for_threshold_mean_std_pt5"

echo ${SLURM_JOB_ID}

srun python find_mean_state_and_state_product_for_threshold.py --input_directory results/ising_model --output_directory results/ising_model --data_subset all --threshold_type mean --threshold 0.5