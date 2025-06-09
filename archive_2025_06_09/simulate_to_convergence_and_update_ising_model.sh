#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --nodelist=hkbugpusrv07
#SBATCH --mem=5G
#SBATCH --gres=gpu:1
#SBATCH --error=results/errors/simulate_to_convergence_and_update_ising_model_sim_10_million_%j.err
#SBATCH --output=results/outs/simulate_to_convergence_and_update_ising_model_sim_10_million_%j.out
#SBATCH --job-name="simulate_to_convergence_and_update_ising_model_sim_10_million"

echo ${SLURM_JOB_ID}

srun python simulate_to_convergence_and_update_ising_model.py --data_directory data --output_directory results/ising_model --data_subset training --threshold median --sim_length 10000000 --num_updates 10 --learning_rate 0.01