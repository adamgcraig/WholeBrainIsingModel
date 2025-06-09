#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --nodelist=hkbugpusrv07
#SBATCH --mem=5G
#SBATCH --gres=gpu:1
#SBATCH --error=results/errors/do_group_correlation_train_test_splits_%j.err
#SBATCH --output=results/outs/do_group_correlation_train_test_splits_%j.out
#SBATCH --job-name="do_group_correlation_train_test_splits"

echo ${SLURM_JOB_ID}

srun python do_group_correlation_train_test_splits.py --data_directory results/ising_model --output_directory results/ising_model --region_feature_file_part node_features_all_as_is --sc_file_part edge_features_all_as_is --group_model_file_part ising_model_light_group_thresholds_31_min_0_max_3_betas_101_min_1e-10_max_0.05_steps_1200_lr_0.01_beta_updates_8_param_updates_40000 --num_permutations 10000 --num_training_regions 180 --num_training_region_pairs 32310