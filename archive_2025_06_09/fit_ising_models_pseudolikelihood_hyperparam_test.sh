#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --nodelist=hkbugpusrv01
#SBATCH --mem=5G
#SBATCH --gres=gpu:1
#SBATCH --error=results/errors/fit_ising_models_pseudolikelihood_hyperparam_test_360_nodes_%j.err
#SBATCH --output=results/outs/fit_ising_models_pseudolikelihood_hyperparam_test_360_nodes_%j.out
#SBATCH --job-name="fit_ising_models_pseudolikelihood_hyperparam_test_360_nodes"

echo ${SLURM_JOB_ID}

srun python fit_ising_models_pseudolikelihood_hyperparam_test.py --data_directory data --output_directory results/ising_model --num_epochs 100000 --num_reps 1000 --num_nodes 360
