#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --nodelist=hkbugpusrv06
#SBATCH --mem=5G
#SBATCH --gres=gpu:1
#SBATCH --error=results/errors/test_ising_model_light_avalanche_counts_%j.err
#SBATCH --output=results/outs/test_ising_model_light_avalanche_counts_%j.out
#SBATCH --job-name="test_ising_light_model_avalanche_counts"

echo ${SLURM_JOB_ID}

srun python test_ising_model_light_avalanche_counts.py --data_directory results/ising_model --output_directory results/ising_model --data_file_name_part all_mean_std_0.5 --model_file_fragment all_mean_std_0.5_medium_init_uncentered_reps_10_target_cov_rmse_lr_0.01_steps_1200_pupd_per_bopt_1000_num_opt_0_bopt_steps_175_popt_steps_7000 --sim_length 120000 --combine_scans