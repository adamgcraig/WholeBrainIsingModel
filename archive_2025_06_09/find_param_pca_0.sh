#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --nodelist=hkbugpusrv02
#SBATCH --mem=5G
#SBATCH --gres=gpu:1
#SBATCH --error=results/errors/find_param_pca_0_%j.err
#SBATCH --output=results/outs/find_param_pca_0_%j.out
#SBATCH --job-name="find_param_pca_0"

echo ${SLURM_JOB_ID}

srun python find_param_pca.py --data_directory results/ising_model --output_directory results/ising_model --iteration_increment 10000 --model_file_name_part ising_model_light_all_mean_std_0_medium_init_uncentered_reps_5_lr_0.01_steps_1200_pupd_per_bopt_1000_num_opt_1_bopt_steps_66_popt_steps_10000 --mean_state_file_name_part mean_state_all_mean_std_0 --mean_state_product_file_name_part mean_state_product_all_mean_std_0 --scale_by_beta