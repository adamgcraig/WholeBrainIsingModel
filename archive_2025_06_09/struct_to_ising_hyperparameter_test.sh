#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --nodelist=hkbugpusrv02
#SBATCH --mem=5G
#SBATCH --gres=gpu:1
#SBATCH --error=results/errors/struct_to_ising_hyperparameter_test_21_nodes_%j.err
#SBATCH --output=results/outs/struct_to_ising_hyperparameter_test_21_nodes_%j.out
#SBATCH --job-name="struct_to_ising_hyperparameter_test_21_nodes"

echo ${SLURM_JOB_ID}

srun python struct_to_ising_hyperparameter_test.py --structural_data_dir data --ising_model_dir results/ising_model --stats_dir results/ising_model --num_epochs 1 --num_subepochs 10000 --validation_batch_size 1000 --num_nodes 21 --num_reps 100 --num_epochs_ising 1000 --window_length 50
