#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --nodelist=hkbugpusrv04
#SBATCH --mem=5G
#SBATCH --gres=gpu:1
#SBATCH --error=results/errors/resume_fitting_ising_model_inv_cov_group_multithreshold_%j.err
#SBATCH --output=results/outs/resume_fitting_ising_model_inv_cov_group_multithreshold_%j.out
#SBATCH --job-name="resume_fitting_ising_model_inv_cov_group_multithreshold"

echo ${SLURM_JOB_ID}

srun python resume_fitting_ising_model_light_pmb.py --data_directory results/ising_model --output_directory results/ising_model --data_file_name_part thresholds_31_min_0_max_3 --output_file_name_part ising_model_light_group_J_inv_cov_h_0_thresh_num_31_min_0_max_3_beta_num_101_min_1e-10_max_1_updates_24_beta_length_120000_param_length_1200_param_updates --updates_per_save 1000 --saves_per_beta_opt 1000 --last_saved_popt 31000