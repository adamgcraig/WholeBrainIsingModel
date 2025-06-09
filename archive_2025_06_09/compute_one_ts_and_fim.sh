#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --nodelist=hkbugpusrv08
#SBATCH --mem=20G
#SBATCH --gres=gpu:1
#SBATCH --error=results/errors/compute_one_ts_and_fim_glasser_group_1_best_%j.err
#SBATCH --output=results/outs/compute_one_ts_and_fim_glasser_group_1_best_%j.out
#SBATCH --job-name="compute_one_ts_and_fim_glasser_group_1_best"

echo ${SLURM_JOB_ID}

srun python compute_one_ts_and_fim.py --data_directory results/ising_model --output_directory results/ising_model --device cuda:2 --model_file_fragment ising_model_light_group_thresholds_31_min_0_max_3_betas_101_min_1e-10_max_0.05_steps_1200_lr_0.01_beta_updates_8_param_updates_63000 --sim_length 64980 --rep_index 89 --target_index 10