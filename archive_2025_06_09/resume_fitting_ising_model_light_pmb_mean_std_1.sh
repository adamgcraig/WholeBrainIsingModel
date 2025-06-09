#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --nodelist=hkbugpusrv02
#SBATCH --mem=5G
#SBATCH --gres=gpu:1
#SBATCH --error=results/errors/resume_fitting_ising_model_light_pmb_mean_std_1_%j.err
#SBATCH --output=results/outs/resume_fitting_ising_model_light_pmb_mean_std_1_%j.out
#SBATCH --job-name="resume_fitting_ising_model_light_pmb_mean_std_1"

echo ${SLURM_JOB_ID}

srun python resume_fitting_ising_model_light_pmb.py --data_directory results/ising_model --output_directory results/ising_model --data_file_name_part all_mean_std_1 --output_file_name_part ising_model_light_all_mean_std_1_medium_init_uncentered_reps_5_lr_0.01_steps_1200_pupd_per_bopt_1000_num_opt_1_bopt_steps_68_popt_steps --updates_per_save 1000 --saves_per_beta_opt 1000 --combine_scans --last_saved_popt 46000 