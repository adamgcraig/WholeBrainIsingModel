#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --nodelist=hkbugpusrv05
#SBATCH --mem=5G
#SBATCH --gres=gpu:1
#SBATCH --error=results/errors/fit_ising_model_individual_all_half_%j.err
#SBATCH --output=results/outs/fit_ising_model_individual_all_half_%j.out
#SBATCH --job-name="fit_ising_model_individual_all_half"

echo ${SLURM_JOB_ID}

srun python fit_ising_model.py --data_directory data --output_directory results/ising_model --model_type individual --data_subset all --num_folds 2 --num_parallel 10 --sim_length 12000 --num_updates_beta 100 --num_updates_boltzmann 1000000 --sims_per_save 500 --learning_rate 0.01 --resume_from_sim 100