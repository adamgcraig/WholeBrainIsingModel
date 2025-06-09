#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --nodelist=hkbugpusrv01
#SBATCH --mem=5G
#SBATCH --gres=gpu:1
#SBATCH --error=results/errors/homogenize_single_param_test_0_myelination_desc_%j.err
#SBATCH --output=results/outs/homogenize_single_param_test_0_myelination_desc_%j.out
#SBATCH --job-name="homogenize_single_param_test_0_myelination_desc"

echo ${SLURM_JOB_ID}

srun python homogenize_single_param_test.py --data_directory results/ising_model --output_directory results/ising_model --data_file_name_part  all_mean_std_0 --model_file_fragment all_mean_std_0_medium_init_uncentered_reps_5_lr_0.01_steps_1200_pupd_per_bopt_1000_num_opt_1_bopt_steps_66_popt_steps_10000 --h_priorities_file h_myelination_corr_all_mean_std_0_medium_init_uncentered_reps_5_lr_0.01_steps_1200_pupd_per_bopt_1000_num_opt_1_bopt_steps_66_popt_steps_10000.pt --combine_scans --homogenize_h --use_abs --descend