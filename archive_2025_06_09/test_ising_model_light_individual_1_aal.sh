#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --nodelist=hkbugpusrv06
#SBATCH --mem=5G
#SBATCH --gres=gpu:1
#SBATCH --error=results/errors/test_ising_model_individual_1_aal_%j.err
#SBATCH --output=results/outs/test_ising_model_individual_1_aal_%j.out
#SBATCH --job-name="fit_ising_model_individual_1_aal"

echo ${SLURM_JOB_ID}

srun python test_ising_model_light_pmb.py --data_directory results/ising_model --output_directory results/ising_model  --combine_scans --data_file_name_part all_aal_mean_std_1 --model_file_fragment all_aal_mean_std_1_medium_init_uncentered_reps_5_beta_min_1e-09_max_0.03_steps_1200_lr_0.01_steps_1200_pupd_per_bopt_1000_num_opt_1_bopt_steps_75_popt_steps_40000 --sim_length 120000