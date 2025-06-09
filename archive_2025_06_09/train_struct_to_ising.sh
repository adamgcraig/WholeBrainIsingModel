#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --nodelist=hkbugpusrv04
#SBATCH --mem=5G
#SBATCH --gres=gpu:1
#SBATCH --error=results/errors/train_struct_to_ising_21_nodes_short_%j.err
#SBATCH --output=results/outs/train_struct_to_ising_21_nodes_short_%j.out
#SBATCH --job-name="train_struct_to_ising_21_nodes_short"

echo ${SLURM_JOB_ID}

srun python train_struct_to_ising.py --structural_data_dir data --ising_model_dir results/ising_model --stats_dir results/ising_model --num_epochs 1 --num_subepochs 500 --validation_batch_size 100 --num_nodes 21 --num_reps 100 --num_epochs_ising 1000 --window_length 50
