#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --nodelist=hkbugpusrv07
#SBATCH --mem=5G
#SBATCH --gres=gpu:1
#SBATCH --error=results/errors/compare_g2gcnn_hyperparameters_big_model_rectangular_max_rmse_2_%j.err
#SBATCH --output=results/outs/compare_g2gcnn_hyperparameters_big_model_rectangular_max_rmse_2_%j.out
#SBATCH --job-name="compare_g2gcnn_hyperparameters_big_model_rectangular_max_rmse_2"

echo ${SLURM_JOB_ID}

srun python compare_g2gcnn_hyperparameters_low_mem.py --input_directory results/g2gcnn_examples --output_directory results/ising_model --file_name_fragment rectangular_max_rmse_2 --output_file_name_fragment big_model --num_epochs 10000 --patience 10 --min_improvement 10e-10 --num_training_examples 3345 --num_validation_examples 420 --num_graph_convolution_layers 10 --min_graph_convolution_layers 2 --max_graph_convolution_layers 20 --num_mlp_hidden_layers 10 --min_mlp_hidden_layers 2 --max_mlp_hidden_layers 20 --num_rep_dims 10 --min_rep_dims 2 --max_rep_dims 20 --num_batch_sizes 3 --min_batch_size 10 --max_batch_size 100 --num_learning_rates 3 --min_learning_rate 0.001 --max_learning_rate 0.1