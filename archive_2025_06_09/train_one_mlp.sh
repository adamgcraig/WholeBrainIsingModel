#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --nodelist=hkbugpusrv03
#SBATCH --mem=5G
#SBATCH --gres=gpu:1
#SBATCH --error=results/errors/train_one_mlp_small_edge_shuffled_%j.err
#SBATCH --output=results/outs/train_one_mlp_small_edge_shuffled_%j.out
#SBATCH --job-name="train_one_mlp_small_edge_shuffled"

echo ${SLURM_JOB_ID}

srun python train_one_mlp.py --input_directory data/ml_examples --output_directory results/ising_model --models_per_subject 5 --file_name_fragment small --optimizer_name Adam --num_epochs 100 --patience 100 --min_improvement 10e-10 --num_instances 5 --mlp_hidden_layers 1 --rep_dims 3 --batch_size 223 --learning_rate 0.001 --is_edge --shuffle_subjects