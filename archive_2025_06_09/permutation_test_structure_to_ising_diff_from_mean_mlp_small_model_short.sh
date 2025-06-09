#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --nodelist=hkbugpusrv04
#SBATCH --mem=5G
#SBATCH --gres=gpu:1
#SBATCH --error=results/errors/permutation_test_structure_to_ising_diff_from_mean_mlp_small_model_short_%j.err
#SBATCH --output=results/outs/permutation_test_structure_to_ising_diff_from_mean_mlp_small_model_short_%j.out
#SBATCH --job-name="permutation_test_structure_to_ising_diff_from_mean_mlp_small_model_short"

echo ${SLURM_JOB_ID}

srun python permutation_test_structure_to_ising_diff_from_mean_mlp.py --input_directory results/ising_model --output_directory results/ising_model --file_name_fragment group_training_and_individual_all --training_index_start 1 --training_index_end 670 --num_permutations 100 --multiply_beta --num_epochs 1000 --learning_rate 0.001 --hidden_layer_width 13 --num_hidden_layers 1