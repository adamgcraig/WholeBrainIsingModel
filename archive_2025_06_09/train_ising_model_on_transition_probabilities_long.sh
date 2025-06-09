#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --nodelist=hkbugpusrv01
#SBATCH --mem=5G
#SBATCH --gres=gpu:1
#SBATCH --error=results/errors/train_ising_model_on_transition_probabilities_%j.err
#SBATCH --output=results/outs/train_ising_model_on_transition_probabilities_%j.out
#SBATCH --job-name="train_ising_model_on_transition_probabilities"

echo ${SLURM_JOB_ID}

srun python train_ising_model_on_transition_probabilities.py --data_directory data --output_directory results/ising_model --data_subset training --num_epochs 1000 --batch_size 25000 --learning_rate 0.0000001