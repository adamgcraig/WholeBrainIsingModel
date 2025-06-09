#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --nodelist=hkbugpusrv07
#SBATCH --mem=5G
#SBATCH --gres=gpu:1
#SBATCH --error=results/errors/select_best_group_ising_model_separate_scans_1_%j.err
#SBATCH --output=results/outs/select_best_group_ising_model_separate_scans_1_%j.out
#SBATCH --job-name="select_best_group_ising_model_separate_scans_1"

echo ${SLURM_JOB_ID}

srun python select_best_group_ising_model.py --input_directory results/ising_model --output_directory results/ising_model --model_file_part ising_model_light_group_thresholds_31_min_0_max_3_betas_101_min_1e-10_max_0.05_steps_1200_lr_0.01_beta_updates_8_param_updates_40000 --fc_corr_file_part fc_corr_ising_model_light_group_thresholds_31_min_0_max_3_betas_101_min_1e-10_max_0.05_steps_1200_lr_0.01_beta_updates_8_param_updates_40000_test_length_120000 --num_thresholds 31 --min_threshold 0.0 --max_threshold 3.0 --target_threshold 1 --num_subjects 3348 --models_per_subject 1