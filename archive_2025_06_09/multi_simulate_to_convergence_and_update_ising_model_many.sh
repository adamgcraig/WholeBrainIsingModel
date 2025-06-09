#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --nodelist=hkbugpusrv04
#SBATCH --mem=5G
#SBATCH --gres=gpu:1
#SBATCH --error=results/errors/multi_simulate_to_convergence_and_update_ising_model_sim_many_%j.err
#SBATCH --output=results/outs/multi_simulate_to_convergence_and_update_ising_model_sim_many_%j.out
#SBATCH --job-name="multi_simulate_to_convergence_and_update_ising_model_sim_many"

echo ${SLURM_JOB_ID}

srun python multi_simulate_to_convergence_and_update_ising_model.py --data_directory data --output_directory results/ising_model --data_subset training --threshold median --num_parallel 10000 --sim_length 1200 --num_updates 10000 --learning_rate 0.001