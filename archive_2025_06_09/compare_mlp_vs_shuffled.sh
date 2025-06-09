#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --nodelist=hkbugpusrv05
#SBATCH --mem=5G
#SBATCH --gres=gpu:1
#SBATCH --error=results/errors/compare_mlp_vs_shuffled_short_%j.err
#SBATCH --output=results/outs/compare_mlp_vs_shuffled_short_%j.out
#SBATCH --job-name="compare_mlp_vs_shuffled_short"

echo ${SLURM_JOB_ID}

srun python compare_mlp_vs_shuffled.py --input_directory results/g2gcnn_examples --output_directory results/ising_model --file_name_fragment rectangular_max_rmse_2 --output_file_name_fragment short --optimizer_name Adam --num_epochs 200 --patience 200 --min_improvement 10e-10 --num_training_examples 3345 --num_validation_examples 420 --num_real 100 --num_shuffled 100 --mlp_hidden_layers 1 --rep_dims 2 --batch_size 669 --learning_rate 0.01