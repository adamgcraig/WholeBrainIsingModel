#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --nodelist=hkbugpusrv02
#SBATCH --mem=5G
#SBATCH --gres=gpu:1
#SBATCH --error=results/errors/test_ising_model_light_mean_std_1_group_10k_individual_10k_%j.err
#SBATCH --output=results/outs/test_ising_model_light_mean_std_1_group_10k_individual_10k_%j.out
#SBATCH --job-name="fit_ising_model_light_mean_std_1_group_10k_individual_10k"

echo ${SLURM_JOB_ID}

srun python test_ising_model_light_pmb.py --data_directory results/ising_model --output_directory results/ising_model --data_file_name_part all_mean_std_1 --model_file_fragment individual_mean_std_1_models_8370_lr_0.01_sim_steps_1.2e+03_plupdates_0_minb_0.008_maxb_0.015_betaopt_10000_simupdates_10000 --sim_length 120000 --combine_scans