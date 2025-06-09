#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --nodelist=hkbugpusrv04
#SBATCH --mem=5G
#SBATCH --gres=gpu:1
#SBATCH --error=results/errors/make_beta_demo_mean_std_0.5_%j.err
#SBATCH --output=results/outs/make_beta_demo_mean_std_0.5_%j.out
#SBATCH --job-name="make_beta_demo_mean_std_0.5"

echo ${SLURM_JOB_ID}

srun python make_beta_demo.py --data_directory results/ising_model --output_directory results/ising_model --data_file_name_part all_mean_std_0.5 --output_file_name_part beta_demo --models_per_subject 10000 --sim_length 120000 --combine_scans --max_beta 1