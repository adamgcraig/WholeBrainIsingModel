#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --nodelist=hkbugpusrv07
#SBATCH --mem=5G
#SBATCH --gres=gpu:1
#SBATCH --error=results/errors/make_fim_demo_%j.err
#SBATCH --output=results/outs/make_fim_demo_%j.out
#SBATCH --job-name="make_fim_demo"

echo ${SLURM_JOB_ID}

srun python make_fim_demo.py --data_directory results/ising_model --output_directory results/ising_model --data_file_name_part all_mean_std_0 --model_file_fragment all_mean_std_0_medium_init_uncentered_reps_5_steps_1200_beta_updates_71_lr_0.01_param_updates_10000 --sim_length 120000 --combine_scans --save_side_length 16245