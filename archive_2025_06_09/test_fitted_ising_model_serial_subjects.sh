#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --nodelist=hkbugpusrv04
#SBATCH --mem=5G
#SBATCH --gres=gpu:1
#SBATCH --error=results/errors/test_fitted_ising_model_serial_subjects_parallel_sims_%j.err
#SBATCH --output=results/outs/test_fitted_ising_model_serial_subjects_parallel_sims_%j.out
#SBATCH --job-name="test_fitted_ising_model_serial_subjects_parallel_sims"

echo ${SLURM_JOB_ID}

srun python test_fitted_ising_model_serial_subjects.py --data_directory data --model_directory results/ising_model --output_directory results/ising_model --data_subset all --expected_num_models 1 --batch_size 1 --ising_param_string parallel_sim_nodes_21_epochs_999_sims_2400_window_50_lr_0.001_threshold_0.100
