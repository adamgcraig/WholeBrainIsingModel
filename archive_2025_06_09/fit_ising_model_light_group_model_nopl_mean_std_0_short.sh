#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --nodelist=hkbugpusrv06
#SBATCH --mem=5G
#SBATCH --gres=gpu:1
#SBATCH --error=results/errors/fit_ising_model_light_group_model_mean_std_0_short_%j.err
#SBATCH --output=results/outs/fit_ising_model_light_group_model_mean_std_0_short_%j.out
#SBATCH --job-name="fit_ising_model_light_group_model_mean_std_0_short"

echo ${SLURM_JOB_ID}

srun python fit_ising_model_light_group_model.py --data_directory results/ising_model --output_directory results/ising_model --threshold 0 --models_per_subject 5 --num_pseudolikelihood_steps 0 --learning_rate 0.01 --block_length 5 --num_beta_opt_steps 1000 --sim_length 1200 --max_beta 1 --num_sim_updates 1000 --verbose