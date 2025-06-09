#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --nodelist=hkbugpusrv04
#SBATCH --mem=5G
#SBATCH --gres=gpu:1
#SBATCH --error=results/errors/permutation_test_structure_to_ising_and_mean_correlations_spherical_%j.err
#SBATCH --output=results/outs/permutation_test_structure_to_ising_and_mean_correlations_spherical_%j.out
#SBATCH --job-name="permutation_test_structure_to_ising_and_mean_correlations_spherical"

echo ${SLURM_JOB_ID}

srun python permutation_test_structure_to_ising_and_mean_correlations.py --input_directory results/ising_model --output_directory results/ising_model --file_name_fragment group_training_and_individual_all --training_index_start 1 --training_index_end 670 --num_permutations 720000 --multiply_beta --spherical_coords