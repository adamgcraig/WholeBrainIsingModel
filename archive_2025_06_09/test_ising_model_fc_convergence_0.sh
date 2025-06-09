#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --nodelist=hkbugpusrv06
#SBATCH --mem=5G
#SBATCH --gres=gpu:1
#SBATCH --error=results/errors/test_ising_model_fc_convergence_0_%j.err
#SBATCH --output=results/outs/test_ising_model_fc_convergence_0_%j.out
#SBATCH --job-name="test_ising_model_fc_convergence_0"

echo ${SLURM_JOB_ID}

srun python test_ising_model_fc_convergence.py --data_directory results/ising_model --output_directory results/ising_model --data_file_name_part all_mean_std_0 --model_file_fragment group_threshold_0_betas_5_min_1e-10_max_0.01_beta_steps_1200_param_steps_1200_lr_0.01_beta_updates_70_param_updates_10000_individual_updates_10000 --min_steps 100 --max_steps 120000 --step_increment 100 --combine_scans --rand_init --device cuda:2