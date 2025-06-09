#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --nodelist=hkbugpusrv04
#SBATCH --mem=5G
#SBATCH --gres=gpu:1
#SBATCH --error=results/errors/permutation_test_ising_model_light_vs_struct_mean_std_1_1m_%j.err
#SBATCH --output=results/outs/permutation_test_ising_model_light_vs_struct_mean_std_1_1m_%j.out
#SBATCH --job-name="permutation_test_ising_model_light_vs_struct_mean_std_1_1m"

echo ${SLURM_JOB_ID}

srun python permutation_test_ising_model_light_vs_struct.py --input_directory results/ising_model --output_directory results/ising_model --feature_file_name_fragment all_as_is --model_file_name_fragment all_mean_std_1_medium_init_uncentered_reps_5_lr_0.01_steps_1200_pupd_per_bopt_1000_num_opt_1_bopt_steps_68_popt_steps_40000 --num_permutations 1000000