#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --nodelist=hkbugpusrv07
#SBATCH --mem=5G
#SBATCH --gres=gpu:1
#SBATCH --error=results/errors/summarize_individual_model_correlations_1_%j.err
#SBATCH --output=results/outs/summarize_individual_model_correlations_1_%j.out
#SBATCH --job-name="summarize_individual_model_correlations_1"

echo ${SLURM_JOB_ID}

srun python summarize_individual_model_correlations.py --data_directory results/ising_model --output_directory results/ising_model --file_name_part ising_model_light_group_threshold_1_betas_5_min_1e-10_max_0.01_beta_steps_1200_param_steps_1200_lr_0.01_beta_updates_68_param_updates_10000_individual_updates_45000