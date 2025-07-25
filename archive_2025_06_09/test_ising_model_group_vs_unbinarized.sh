#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --nodelist=hkbugpusrv06
#SBATCH --mem=5G
#SBATCH --gres=gpu:1
#SBATCH --error=results/errors/test_ising_model_group_multithreshold_vs_as_is_%j.err
#SBATCH --output=results/outs/test_ising_model_group_multithreshold_vs_as_is_%j.out
#SBATCH --job-name="fit_ising_model_group_multithreshold_vs_as_is"

echo ${SLURM_JOB_ID}

srun python test_ising_model_light_pmb.py --data_directory results/ising_model --output_directory results/ising_model --data_file_name_part group_as_is --model_file_fragment group_thresholds_31_min_0_max_3_betas_101_min_1e-10_max_0.05_steps_1200_lr_0.01_beta_updates_8_param_updates_40000 --sim_length 120000