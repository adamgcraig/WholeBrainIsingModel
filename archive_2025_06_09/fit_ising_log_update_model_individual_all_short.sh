#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --nodelist=hkbugpusrv01
#SBATCH --mem=5G
#SBATCH --gres=gpu:1
#SBATCH --error=results/errors/fit_ising_model_log_update_individual_all_whole_short_%j.err
#SBATCH --output=results/outs/fit_ising_model_log_update_individual_all_whole_short_%j.out
#SBATCH --job-name="fit_ising_model_log_update_individual_all_whole_short"

echo ${SLURM_JOB_ID}

srun python fit_ising_model_log_update.py --data_directory data --output_directory results/ising_model --model_type individual --data_subset all --num_folds 1 --num_parallel 10 --sim_length 1200 --num_updates_beta 100 --num_updates_boltzmann 1000000 --sims_per_save 2500 --learning_rate 0.01