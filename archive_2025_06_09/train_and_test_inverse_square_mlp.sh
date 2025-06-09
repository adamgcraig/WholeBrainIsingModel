#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --nodelist=hkbugpusrv03
#SBATCH --mem=5G
#SBATCH --gres=gpu:1
#SBATCH --error=results/errors/train_and_test_inverse_square_mlp_smaller_%j.err
#SBATCH --output=results/outs/train_and_test_inverse_square_mlp_smaller_%j.out
#SBATCH --job-name="train_and_test_inverse_square_mlp_smaller"

echo ${SLURM_JOB_ID}

srun python train_and_test_inverse_square_mlp.py --input_directory results/ising_model --output_directory results/ising_model --file_name_fragment group_training_and_individual_all --training_index_start 1 --training_index_end 670 --validation_index_start 670 --validation_index_end 754 --num_permutations 100 --multiply_beta --num_epochs 10000 --learning_rate 0.001 --hidden_layer_width 13 --num_hidden_layers 1 --num_embedding_dims 3 --sim_length 120000