#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --nodelist=hkbugpusrv06
#SBATCH --mem=5G
#SBATCH --gres=gpu:1
#SBATCH --error=results/errors/resume_fitting_ising_model_light_pmb_mean_std_1_from_group_20k_%j.err
#SBATCH --output=results/outs/resume_fitting_ising_model_light_pmb_mean_std_1_from_group_20k_%j.out
#SBATCH --job-name="resume_fitting_ising_model_light_pmb_mean_std_1_from_group_20k"

echo ${SLURM_JOB_ID}

srun python resume_fitting_ising_model_light_pmb.py --data_directory results/ising_model --output_directory results/ising_model --data_file_name_part all_mean_std_1 --output_file_name_part light_group_threshold_1_betas_5_min_1e-10_max_0.01_beta_steps_1200_param_steps_1200_lr_0.01_beta_updates_68_param_updates_20000_individual_updates --updates_per_save 1000 --saves_per_beta_opt 1000 --combine_scans --last_saved_popt 13000 