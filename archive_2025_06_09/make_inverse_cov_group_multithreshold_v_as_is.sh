#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --nodelist=hkbugpusrv01
#SBATCH --mem=5G
#SBATCH --gres=gpu:1
#SBATCH --error=results/errors/make_inverse_cov_group_multithreshold_v_as_is_%j.err
#SBATCH --output=results/outs/make_inverse_cov_group_multithreshold_v_as_is_%j.out
#SBATCH --job-name="make_inverse_cov_group_multithreshold_v_as_is"

echo ${SLURM_JOB_ID}

srun python make_inverse_cov_group_multithreshold.py --input_directory results/ising_model --output_directory results/ising_model --mean_state_file_part mean_state_all_as_is --mean_state_product_file_part mean_state_product_all_as_is --as_is --model_file_part ising_model_light_group_thresholds_31_min_0_max_3_betas_101_min_1e-10_max_0.05_steps_1200_lr_0.01_beta_updates_8_param_updates_40000