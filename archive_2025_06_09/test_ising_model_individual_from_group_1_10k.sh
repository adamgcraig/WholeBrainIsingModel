#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --nodelist=hkbugpusrv07
#SBATCH --mem=5G
#SBATCH --gres=gpu:1
#SBATCH --error=results/errors/test_ising_model_individual_from_group_1_10k_%j.err
#SBATCH --output=results/outs/test_ising_model_individual_from_group_1_10k_%j.out
#SBATCH --job-name="test_ising_model_individual_from_group_1_10k"

echo ${SLURM_JOB_ID}

srun python test_ising_model_light_series.py --data_directory results/ising_model --output_directory results/ising_model  --data_file_name_part all_mean_std_1 --model_file_fragment light_group_threshold_1_betas_5_min_1e-10_max_0.01_beta_steps_1200_param_steps_1200_lr_0.01_beta_updates_68_param_updates_10000_individual_updates --min_updates 29000 --max_updates 38000