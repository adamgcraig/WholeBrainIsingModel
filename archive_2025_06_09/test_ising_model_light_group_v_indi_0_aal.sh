#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --nodelist=hkbugpusrv05
#SBATCH --mem=5G
#SBATCH --gres=gpu:1
#SBATCH --error=results/errors/test_ising_model_group_v_indi_0_aal_%j.err
#SBATCH --output=results/outs/test_ising_model_group_v_indi_0_aal_%j.out
#SBATCH --job-name="fit_ising_model_group_v_indi_0_aal"

echo ${SLURM_JOB_ID}

srun python test_ising_model_light_pmb.py --data_directory results/ising_model --output_directory results/ising_model  --combine_scans --data_file_name_part all_aal_mean_std_0 --model_file_fragment group_init_means_thresholds_31_min_0_max_3_betas_101_min_1e-10_max_1_steps_1200_lr_0.01_beta_updates_8_v2_param_updates_40000_to_thresh_0_reps_5_subj_837 --sim_length 120000