#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --nodelist=hkbugpusrv05
#SBATCH --mem=5G
#SBATCH --gres=gpu:1
#SBATCH --error=results/errors/test_ising_model_light_point_process_stats_mean_std_0_medium_%j.err
#SBATCH --output=results/outs/test_ising_model_light_point_process_stats_mean_std_0_medium_%j.out
#SBATCH --job-name="test_ising_model_light_point_process_stats_mean_std_0_medium"

echo ${SLURM_JOB_ID}
#     --max_samples_at_once 1000
srun python test_ising_model_light_point_process_stats.py --data_directory results/ising_model --output_directory results/ising_model --data_file_name_part all_mean_std_0 --model_file_fragment all_mean_std_0_medium_init_uncentered_reps_5_lr_0.01_steps_1200_pupd_per_bopt_1000_num_opt_1_bopt_steps_66_popt_steps_10000 --sim_length 1200 --combine_scans --num_passes 1000 --num_tries_per_pass 1000 --num_beta_multiplier_passes 1000 --num_beta_multipliers_per_pass 40 --min_beta_multiplier 1.0 --max_beta_multiplier 20.0 --num_p_value_distances 1000000