#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --nodelist=hkbugpusrv07
#SBATCH --mem=5G
#SBATCH --gres=gpu:1
#SBATCH --error=results/errors/resume_fitting_ising_model_inv_cov_group_multithreshold_multi_lr_%j.err
#SBATCH --output=results/outs/resume_fitting_ising_model_inv_cov_group_multithreshold_multi_lr_%j.out
#SBATCH --job-name="resume_fitting_ising_model_inv_cov_group_multithreshold_multi_lr"

echo ${SLURM_JOB_ID}

srun python resume_fitting_ising_model_light_pmb_multi_lr.py --data_directory results/ising_model --output_directory results/ising_model --data_file_name_part thresholds_31_min_0_max_3 --output_file_name_part ising_model_group_J_inv_cov_h_0_thresholds_31_min_0_max_3_beta_num_101_min_1e-10_max_1_sim_length_12000_updates_8_param_sim_length_1200_updates --updates_per_save 1000 --saves_per_beta_opt 1000 --last_saved_popt 9000