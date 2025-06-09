#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --nodelist=hkbugpusrv07
#SBATCH --mem=5G
#SBATCH --gres=gpu:1
#SBATCH --error=results/errors/fit_group_ising_model_pseudolikelihood_medium_%j.err
#SBATCH --output=results/outs/fit_group_ising_model_pseudolikelihood_medium_%j.out
#SBATCH --job-name="fit_group_ising_model_pseudolikelihood_medium"

echo ${SLURM_JOB_ID}

srun python fit_group_ising_model_pseudolikelihood.py --data_directory results/ising_model --output_directory results/ising_model --data_file_fragment mean_std_1 --num_inits 5 --num_pl_updates 20000 --pl_learning_rate 0.01 --print_every_steps 200 --num_betas 837 --num_beta_updates 1000 --beta_sim_length 12000 --num_param_updates 1000 --param_sim_length 1200 --param_learning_rate 0.01 --num_saves 1000