#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --nodelist=hkbugpusrv05
#SBATCH --mem=5G
#SBATCH --gres=gpu:1
#SBATCH --error=results/errors/test_ising_model_light_mean_std_0_individual_10k_%j.err
#SBATCH --output=results/outs/test_ising_model_light_mean_std_0_individual_10k_%j.out
#SBATCH --job-name="fit_ising_model_light_mean_std_0_individual_10k"

echo ${SLURM_JOB_ID}

srun python test_ising_model_light_pmb.py --data_directory results/ising_model --output_directory results/ising_model --data_file_name_part all_mean_std_0 --model_file_fragment all_mean_std_0_medium_init_uncentered_reps_5_lr_0.01_steps_1200_pupd_per_bopt_1000_num_opt_1_bopt_steps_66_popt_steps_10000 --sim_length 120000 --combine_scans