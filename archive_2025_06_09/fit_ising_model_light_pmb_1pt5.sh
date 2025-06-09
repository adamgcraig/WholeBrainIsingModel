#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --nodelist=hkbugpusrv08
#SBATCH --mem=5G
#SBATCH --gres=gpu:1
#SBATCH --error=results/errors/fit_ising_model_light_pmb_medium_mean_std_1pt5_%j.err
#SBATCH --output=results/outs/fit_ising_model_light_pmb_medium_mean_std_1pt5_%j.out
#SBATCH --job-name="fit_ising_model_light_pmb_medium_mean_std_1pt5"

echo ${SLURM_JOB_ID}

srun python fit_ising_model_light_pmb.py --data_directory results/ising_model --output_directory results/ising_model --data_file_name_part all_mean_std_1.5 --output_file_name_part medium --models_per_subject 5 --num_updates_beta 1000000 --updates_per_save 1000 --saves_per_beta_opt 1000 --num_beta_opts 1 --combine_scans --max_beta 0.0065