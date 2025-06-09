#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --nodelist=hkbugpusrv07
#SBATCH --mem=5G
#SBATCH --gres=gpu:1
#SBATCH --error=results/errors/find_group_ising_fisher_information_eigs_offsets_%j.err
#SBATCH --output=results/outs/find_group_ising_fisher_information_eigs_offsets_%j.out
#SBATCH --job-name="find_group_ising_fisher_information_eigs_offsets"

echo ${SLURM_JOB_ID}

srun python find_group_ising_fisher_information_eigs_offsets.py --data_dir data --model_dir results/ising_model --model_dir results/ising_model --num_nodes 21 --model_param_string nodes_21_window_50_lr_0.000_threshold_0.100_beta_0.500 --num_reps_indi 100 --epoch_indi 1999 --num_reps_group 100 --epoch_group 6