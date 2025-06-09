#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --nodelist=hkbugpusrv05
#SBATCH --mem=5G
#SBATCH --gres=gpu:1
#SBATCH --error=results/errors/individual_replica_structure_parameter_correlations_glasser_1_%j.err
#SBATCH --output=results/outs/individual_replica_structure_parameter_correlations_glasser_1_%j.out
#SBATCH --job-name="individual_replica_structure_parameter_correlations_glasser_1"

echo ${SLURM_JOB_ID}

srun python individual_replica_structure_parameter_correlations.py --data_directory results/ising_model --output_directory results/ising_model --individual_fmri_file_name_part all_mean_std_1 --individual_model_file_part ising_model_light_group_thresholds_31_min_0_max_3_betas_101_min_1e-10_max_0.05_steps_1200_lr_0.01_beta_updates_8_param_updates_40000_to_thresh_1_reps_5_subj_837_individual_updates_40000 --individual_model_short_identifier individual_from_group_glasser_1 --threshold_index 10