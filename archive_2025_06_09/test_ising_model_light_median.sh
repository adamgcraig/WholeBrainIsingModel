#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --nodelist=hkbugpusrv08
#SBATCH --mem=5G
#SBATCH --gres=gpu:1
#SBATCH --error=results/errors/test_ising_model_light_median_%j.err
#SBATCH --output=results/outs/test_ising_model_light_median_%j.out
#SBATCH --job-name="test_ising_model_light_median"

echo ${SLURM_JOB_ID}

srun python test_ising_model_light.py --data_directory results/ising_model --output_directory results/ising_model --data_file_name_part all_quantile_0.5 --model_file_fragment all_quantile_0.5_medium_init_uncentered_reps_10_steps_1200_beta_updates_31_lr_0.01_param_updates_3000 --sim_length 120000 --combine_scans