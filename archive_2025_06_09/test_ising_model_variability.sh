#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --nodelist=hkbugpusrv03
#SBATCH --mem=5G
#SBATCH --gres=gpu:1
#SBATCH --error=results/errors/test_model_variability_21_nodes_training_%j.err
#SBATCH --output=results/outs/test_model_variability_21_nodes_training_%j.out
#SBATCH --job-name="test_model_variability_21_nodes_training"

echo ${SLURM_JOB_ID}

srun python test_ising_model_variability.py --data_directory data --model_directory results/ising_model --output_directory results/ising_model --data_subset training --num_nodes 21 --sim_length 48000
