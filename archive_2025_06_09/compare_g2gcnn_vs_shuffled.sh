#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --nodelist=hkbugpusrv01
#SBATCH --mem=5G
#SBATCH --gres=gpu:1
#SBATCH --error=results/errors/compare_g2gcnn_vs_shuffled_%j.err
#SBATCH --output=results/outs/compare_g2gcnn_vs_shuffled_%j.out
#SBATCH --job-name="compare_g2gcnn_vs_shuffled"

echo ${SLURM_JOB_ID}

srun python compare_g2gcnn_vs_shuffled.py --input_directory results/g2gcnn_examples --output_directory results/ising_model --file_name_fragment rectangular_max_rmse_2 --output_file_name_fragment first --optimizer_name Adam --num_epochs 10000 --patience 10 --min_improvement 10e-10 --num_training_examples 3345 --num_validation_examples 420 --num_real 100 --num_shuffled 100 --graph_convolution_layers 3 --mlp_hidden_layers 3 --rep_dims 7 --batch_size 100 --learning_rate 0.001