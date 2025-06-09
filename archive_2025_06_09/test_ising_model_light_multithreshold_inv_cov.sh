#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --nodelist=hkbugpusrv06
#SBATCH --mem=5G
#SBATCH --gres=gpu:1
#SBATCH --error=results/errors/test_ising_model_group_multithreshold_inv_cov_%j.err
#SBATCH --output=results/outs/test_ising_model_group_multithreshold_inv_cov_%j.out
#SBATCH --job-name="fit_ising_model_group_multithreshold_inv_cov"

echo ${SLURM_JOB_ID}

srun python test_ising_model_light_pmb.py --data_directory results/ising_model --output_directory results/ising_model --data_file_name_part thresholds_31_min_0_max_3 --init_params_file_name_part thresholds_31_min_0_max_3 --model_file_fragment group_J_inv_cov_h_0_thresh_num_31_min_0_max_3_beta_num_101_min_1e-10_max_1_updates_24_beta_length_120000_param_length_1200_param_updates_30000 --sim_length 120000