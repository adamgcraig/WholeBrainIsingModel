#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --nodelist=hkbugpusrv01
#SBATCH --mem=5G
#SBATCH --gres=gpu:1
#SBATCH --error=results/errors/individualize_ising_model_pmb_0_1k_%j.err
#SBATCH --output=results/outs/individualize_ising_model_pmb_0_1k_%j.out
#SBATCH --job-name="individualize_ising_model_pmb_0_1k"

echo ${SLURM_JOB_ID}

srun python individualize_ising_model_pmb.py --input_directory results/ising_model --output_directory results/ising_model --group_model_file_fragment light_group_threshold_0_betas_5_min_1e-10_max_0.01_beta_steps_1200_param_steps_1200_lr_0.01_beta_updates_70_param_updates_1000 --threshold_z 0 --param_sim_length 1200 --updates_per_save 1000 --num_saves 20 --learning_rate 0.01