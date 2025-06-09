#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --nodelist=hkbugpusrv06
#SBATCH --mem=5G
#SBATCH --gres=gpu:1
#SBATCH --error=results/errors/find_structure_to_ising_and_mean_correlations_%j.err
#SBATCH --output=results/outs/find_structure_to_ising_and_mean_correlations_%j.out
#SBATCH --job-name="find_structure_to_ising_and_mean_correlations"

echo ${SLURM_JOB_ID}

srun python find_structure_to_ising_and_mean_correlations.py --input_directory results/ising_model --output_directory results/ising_model --file_name_fragment group_training_and_individual_all --training_index_start 1 --training_index_end 670