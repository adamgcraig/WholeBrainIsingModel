#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --nodelist=hkbugpusrv04
#SBATCH --mem=5G
#SBATCH --gres=gpu:1
#SBATCH --error=results/errors/individualize_ising_model_glasser_1_%j.err
#SBATCH --output=results/outs/individualize_ising_model_glasser_1_%j.out
#SBATCH --job-name="individualize_ising_model_glasser_1"

echo ${SLURM_JOB_ID}

srun python individualize_ising_model_pmb.py --input_directory results/ising_model --output_directory results/ising_model --target_data_file_fragment all_mean_std_1 --group_model_file_fragment ising_model_light_group_thresholds_31_min_0_max_3_betas_101_min_1e-10_max_0.05_steps_1200_lr_0.01_beta_updates_8_param_updates_40000_to_thresh_1_reps_5_subj_837 --param_sim_length 1200 --updates_per_save 1000 --num_saves 1000 --learning_rate 0.01