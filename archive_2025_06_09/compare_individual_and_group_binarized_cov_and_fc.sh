#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --nodelist=hkbugpusrv08
#SBATCH --mem=5G
#SBATCH --gres=gpu:1
#SBATCH --error=results/errors/compare_individual_and_group_binarized_cov_and_fc_%j.err
#SBATCH --output=results/outs/compare_individual_and_group_binarized_cov_and_fc_%j.out
#SBATCH --job-name="compare_individual_and_group_binarized_cov_and_fc"

echo ${SLURM_JOB_ID}

srun python compare_individual_and_group_binarized_cov_and_fc.py --input_directory results/ising_model --output_directory results/ising_model --num_thresholds 121