#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --nodelist=hkbugpusrv05
#SBATCH --mem=5G
#SBATCH --gres=gpu:1
#SBATCH --error=results/errors/resume_fitting_ising_model_from_group_glasser_1_single_scan_first_%j.err
#SBATCH --output=results/outs/resume_fitting_ising_model_from_group_glasser_1_single_scan_first_%j.out
#SBATCH --job-name="resume_fitting_ising_model_from_group_glasser_1_single_scan_first"

echo ${SLURM_JOB_ID}

srun python resume_fitting_ising_model_light_pmb.py --data_directory results/ising_model --output_directory results/ising_model --data_file_name_part all_mean_std_1 --output_file_name_part ising_model_light_group_thresholds_31_min_0_max_3_betas_101_min_1e-10_max_0.05_steps_1200_lr_0.01_beta_updates_8_param_updates_40000_to_thresh_1_reps_1_subj_3348_v1_individual_updates --updates_per_save 1000 --saves_per_beta_opt 1000 --last_saved_popt 51000