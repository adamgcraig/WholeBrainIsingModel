#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --nodelist=hkbugpusrv02
#SBATCH --mem=5G
#SBATCH --gres=gpu:1
#SBATCH --error=results/errors/permutation_test_ising_model_light_vs_struct_mean_std_0.5_10k_%j.err
#SBATCH --output=results/outs/permutation_test_ising_model_light_vs_struct_mean_std_0.5_10k_%j.out
#SBATCH --job-name="permutation_test_ising_model_light_vs_struct_mean_std_0.5_10k"

echo ${SLURM_JOB_ID}

srun python permutation_test_ising_model_light_vs_struct.py --input_directory results/ising_model --output_directory results/ising_model --feature_file_name_fragment all_as_is --model_file_name_fragment all_mean_std_0.5_medium_init_uncentered_reps_10_target_cov_rmse_lr_0.01_steps_1200_pupd_per_bopt_1000_num_opt_0_bopt_steps_175_popt_steps_10000 --num_permutations 100000