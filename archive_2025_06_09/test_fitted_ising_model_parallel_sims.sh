#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --nodelist=hkbugpusrv01
#SBATCH --mem=5G
#SBATCH --gres=gpu:1
#SBATCH --error=results/errors/test_fitted_ising_model_parallel_sims_%j.err
#SBATCH --output=results/outs/test_fitted_ising_model_parallel_sims_%j.out
#SBATCH --job-name="test_fitted_ising_model_parallel_sims"

echo ${SLURM_JOB_ID}

srun python test_fitted_ising_model_parallel_sims.py --data_directory data --model_directory results/ising_model --output_directory results/ising_model --subject_id 516742 --num_sims 1000 --batch_size 1000 --ising_param_string _parallel_nodes_21_epochs_1000_reps_100_window_50_lr_0.001_threshold_0.100_beta_0.500
