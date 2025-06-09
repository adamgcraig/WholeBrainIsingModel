#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --nodelist=hkbugpusrv03
#SBATCH --mem=5G
#SBATCH --gres=gpu:1
#SBATCH --error=results/errors/individualize_ising_model_%j.err
#SBATCH --output=results/outs/individualize_ising_model_%j.out
#SBATCH --job-name="individualize_ising_model"

echo ${SLURM_JOB_ID}

srun python individualize_ising_model.py --data_directory data --output_directory results/ising_model --data_subset training --num_parallel 10000 --sim_length 120000 --num_updates_beta 3 --num_updates 1000 --sims_per_save 10 --learning_rate 0.1