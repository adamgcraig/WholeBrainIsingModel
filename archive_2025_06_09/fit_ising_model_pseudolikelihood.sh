#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --nodelist=hkbugpusrv04
#SBATCH --mem=5G
#SBATCH --gres=gpu:1
#SBATCH --error=results/errors/fit_ising_model_pseudolikelihood_slow_%j.err
#SBATCH --output=results/outs/fit_ising_model_pseudolikelihood_slow_%j.out
#SBATCH --job-name="fit_ising_model_pseudolikelihood_slow"

echo ${SLURM_JOB_ID}

srun python fit_ising_model_pseudolikelihood.py --data_directory data --output_directory results/ising_model --data_set individual_all --num_folds 1 --num_betas_per_target 1 --learning_rate 0.002 --num_param_saves 50 --param_updates_per_save 100000