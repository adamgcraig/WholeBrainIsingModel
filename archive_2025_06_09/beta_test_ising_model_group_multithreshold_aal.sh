#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --nodelist=hkbugpusrv06
#SBATCH --mem=5G
#SBATCH --gres=gpu:1
#SBATCH --error=results/errors/beta_test_ising_model_multithreshold_aal_%j.err
#SBATCH --output=results/outs/beta_test_ising_model_multithreshold_aal_%j.out
#SBATCH --job-name="beta_test_ising_model_multithreshold_aal"

echo ${SLURM_JOB_ID}

srun python beta_test_ising_model_pmb.py --data_directory results/ising_model --output_directory results/ising_model --data_file_name_part aal_thresholds_31_min_0_max_3 --model_file_fragment light_group_init_means_thresholds_31_min_0_max_3_betas_101_min_1e-10_max_1_steps_1200_lr_0.01_beta_updates_8 --sim_length 120000 --max_beta 0.05