#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --nodelist=hkbugpusrv08
#SBATCH --mem=5G
#SBATCH --gres=gpu:1
#SBATCH --error=results/errors/fit_ising_model_light_medium_as_is_%j.err
#SBATCH --output=results/outs/fit_ising_model_light_medium_as_is_%j.out
#SBATCH --job-name="fit_ising_model_light_medium_as_is"

echo ${SLURM_JOB_ID}

srun python fit_ising_model_light.py --data_directory results/ising_model --output_directory results/ising_model --data_file_name_part all_as_is --output_file_name_part medium --models_per_subject 5 --sim_length 1200 --num_updates_beta 10000 --updates_per_save 1000 --num_saves 10 --learning_rate 0.01 --max_beta 100 --combine_scans