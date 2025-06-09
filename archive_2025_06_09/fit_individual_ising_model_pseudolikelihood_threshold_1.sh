#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --nodelist=hkbugpusrv03
#SBATCH --mem=5G
#SBATCH --gres=gpu:1
#SBATCH --error=results/errors/fit_individual_ising_model_pseudolikelihood_threshold_1_short_%j.err
#SBATCH --output=results/outs/fit_individual_ising_model_pseudolikelihood_threshold_1_short_%j.out
#SBATCH --job-name="fit_individual_ising_model_pseudolikelihood_threshold_1_short"

echo ${SLURM_JOB_ID}

srun python fit_individual_ising_model_pseudolikelihood.py --data_directory results/ising_model --output_directory results/ising_model --threshold 1.0 --learning_rate 0.01 --num_saves 20 --updates_per_save 1000 --num_betas 11 --max_beta 1.0 --sim_length 120000