#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --nodelist=hkbugpusrv06
#SBATCH --mem=5G
#SBATCH --gres=gpu:1
#SBATCH --error=results/errors/find_structure_to_ising_correlations_%j.err
#SBATCH --output=results/outs/find_structure_to_ising_correlations_%j.out
#SBATCH --job-name="find_structure_to_ising_correlations"

echo ${SLURM_JOB_ID}

srun python find_structure_to_ising_correlations.py --input_directory results/g2gcnn_examples --output_directory results/ising_model --file_name_fragment rectangular_max_rmse_2 --num_training_examples 3345 --batch_size 100