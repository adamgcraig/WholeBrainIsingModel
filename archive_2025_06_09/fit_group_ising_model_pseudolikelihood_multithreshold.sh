#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --nodelist=hkbugpusrv08
#SBATCH --mem=5G
#SBATCH --gres=gpu:1
#SBATCH --error=results/errors/fit_group_ising_model_pseudolikelihood_multithreshold_long_%j.err
#SBATCH --output=results/outs/fit_group_ising_model_pseudolikelihood_multithreshold_long_%j.out
#SBATCH --job-name="fit_group_ising_model_pseudolikelihood_multithreshold_long"

echo ${SLURM_JOB_ID}

srun python fit_group_ising_model_pseudolikelihood_multithreshold.py --data_directory results/ising_model --output_directory results/ising_model --num_thresholds 3 --min_threshold 0 --max_threshold 2.4 --learning_rate 0.01 --num_saves 400 --steps_per_save 1000 --num_betas 101 --max_beta 1.0 --sim_length 120000