#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --nodelist=hkbugpusrv01
#SBATCH --mem=5G
#SBATCH --gres=gpu:1
#SBATCH --error=results/errors/test_ising_model_group_reset_multithreshold_%j.err
#SBATCH --output=results/outs/test_ising_model_group_reset_multithreshold_%j.out
#SBATCH --job-name="fit_ising_model_group_reset_multithreshold"

echo ${SLURM_JOB_ID}

srun python test_ising_model_light_pmb.py --data_directory results/ising_model --output_directory results/ising_model  --combine_scans --data_file_name_part thresholds_31_min_0_max_3 --init_params_file_name_part thresholds_31_min_0_max_3 --model_file_fragment group_thresholds_31_min_0_max_3_betas_101_min_1e-10_max_0.05_steps_1200_lr_0.01_beta_updates_8_param_updates_54000 --sim_length 120000 --reset_params