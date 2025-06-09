#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --nodelist=hkbugpusrv01
#SBATCH --mem=5G
#SBATCH --gres=gpu:1
#SBATCH --time=14-00:00
#SBATCH --error=results/errors/fit_ising_models_batch_lowmem_training_50_epochs_%j.err
#SBATCH --output=results/outs/fit_ising_models_batch_daai_lowmem_training_50_epochs_%j.out
#SBATCH --job-name="fit_ising_models_batch_daai_lowmem_training_50_epochs"

echo ${SLURM_JOB_ID}

srun python fit_ising_models_batch_daai_lowmem.py --num_nodes 21 --num_steps 50 --num_reps 10 --subjects_start 0 --subjects_end 670 --data_subset training --num_epochs 50