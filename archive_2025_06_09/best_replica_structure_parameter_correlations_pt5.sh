#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --nodelist=hkbugpusrv01
#SBATCH --mem=5G
#SBATCH --gres=gpu:1
#SBATCH --error=results/errors/best_replica_structure_parameter_correlations_glasser_pt5_%j.err
#SBATCH --output=results/outs/best_replica_structure_parameter_correlations_glasser_pt5_%j.out
#SBATCH --job-name="best_replica_structure_parameter_correlations_glasser_pt5"

echo ${SLURM_JOB_ID}

srun python best_replica_structure_parameter_correlations.py --data_directory results/ising_model --output_directory results/ising_model --individual_fmri_file_name_part all_mean_std_0.5 --individual_model_goodness_file_part fc_corr_group_thresholds_31_min_0_max_3_betas_101_min_1e-10_max_0.05_steps_1200_lr_0.01_beta_updates_8_param_updates_40000_to_thresh_1_reps_5_subj_837_individual_updates_40000_test_length_120000 --individual_model_file_part ising_model_light_group_thresholds_31_min_0_max_3_betas_101_min_1e-10_max_0.05_steps_1200_lr_0.01_beta_updates_8_param_updates_40000_to_thresh_0.5_reps_5_subj_837_individual_updates_40000 --individual_model_short_identifier individual_from_group_glasser_pt5 --threshold_index 5 --num_perms_group_node 10000 --num_perms_group_pair 10000 --num_perms_individual_node 10000 --num_perms_individual_pair 10000 --num_perms_train_test_node 10000 --num_perms_train_test_pair 10000