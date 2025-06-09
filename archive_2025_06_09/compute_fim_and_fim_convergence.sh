#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --nodelist=hkbugpusrv03
#SBATCH --mem=5G
#SBATCH --gres=gpu:1
#SBATCH --error=results/errors/compute_fim_and_fim_convergence_group_training_%j.err
#SBATCH --output=results/outs/compute_fim_and_fim_convergence_group_training_%j.out
#SBATCH --job-name="compute_fim_and_fim_convergence_group_training"

echo ${SLURM_JOB_ID}

srun python compute_fim_and_fim_convergence.py --data_directory data --output_directory results/ising_model --model_set group_training --threshold 0.1 --num_nodes 21 --reps_per_batch 1000 --num_batches 1 --window_length 50 --num_epochs 34 --test_length 480000
