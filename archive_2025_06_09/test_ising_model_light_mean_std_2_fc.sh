#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --nodelist=hkbugpusrv05
#SBATCH --mem=5G
#SBATCH --gres=gpu:1
#SBATCH --error=results/errors/test_ising_model_light_mean_std_2_fc_%j.err
#SBATCH --output=results/outs/test_ising_model_light_mean_std_2_fc_%j.out
#SBATCH --job-name="fit_ising_model_light_mean_std_2_fc"

echo ${SLURM_JOB_ID}

srun python test_ising_model_light.py --data_directory results/ising_model --output_directory results/ising_model --data_file_name_part all_mean_std_2 --model_file_fragment all_mean_std_2_long_init_uncentered_reps_5_steps_12000_beta_updates_141_lr_0.01_param_updates_1000 --sim_length 120000 --combine_scans --reset_params