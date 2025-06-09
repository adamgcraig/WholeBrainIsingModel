#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --nodelist=hkbugpusrv01
#SBATCH --mem=5G
#SBATCH --gres=gpu:1
#SBATCH --error=results/errors/compare_mlp_hyperparameters_deep_%j.err
#SBATCH --output=results/outs/compare_mlp_hyperparameters_deep_%j.out
#SBATCH --job-name="compare_mlp_hyperparameters_deep"

echo ${SLURM_JOB_ID}

srun python compare_mlp_hyperparameters.py --input_directory results/g2gcnn_examples --output_directory results/ising_model --file_name_fragment rectangular_max_rmse_2 --output_file_name_fragment deep --num_epochs 10000 --patience 1000 --min_improvement 10e-10 --num_training_examples 3345 --num_validation_examples 420 --num_mlp_hidden_layers 3 --min_mlp_hidden_layers 10 --max_mlp_hidden_layers 100 --num_rep_dims 3 --min_rep_dims 20 --max_rep_dims 1000 --num_batch_sizes 3 --min_batch_size 1 --max_batch_size 100 --num_learning_rates 3 --min_learning_rate 0.01 --max_learning_rate 1.0