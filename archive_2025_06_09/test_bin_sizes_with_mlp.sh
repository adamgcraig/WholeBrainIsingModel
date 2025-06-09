#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --nodelist=hkbugpusrv03
#SBATCH --mem=5G
#SBATCH --gres=gpu:1
#SBATCH --error=results/errors/test_bin_sizes_with_mlp_%j.err
#SBATCH --output=results/outs/test_bin_sizes_with_mlp_%j.out
#SBATCH --job-name="test_bin_sizes_with_mlp"

echo ${SLURM_JOB_ID}

srun python test_bin_sizes_with_mlp.py --input_directory results/ising_model --output_directory results/ising_model --file_name_fragment group_training_and_individual_all --training_index_start 1 --training_index_end 670 --validation_index_start 670 --validation_index_end 754 --num_permutations 10 --multiply_beta --num_epochs 1000 --learning_rate 0.001 --hidden_layer_width 13 --num_hidden_layers 1 --neighborhood_in_std_devs_increment 0.1