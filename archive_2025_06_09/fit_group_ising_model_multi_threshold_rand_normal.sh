#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --nodelist=hkbugpusrv01
#SBATCH --mem=5G
#SBATCH --gres=gpu:1
#SBATCH --error=results/errors/fit_group_ising_model_multi_threshold_rand_normal_%j.err
#SBATCH --output=results/outs/fit_group_ising_model_multi_threshold_rand_normal_%j.out
#SBATCH --job-name="fit_group_ising_model_multi_threshold_rand_normal"

echo ${SLURM_JOB_ID}

srun python fit_group_ising_model_multi_threshold.py --input_directory results/ising_model --output_directory results/ising_model --num_thresholds 31 --min_threshold 0  --max_threshold 3 --models_per_threshold 101 --max_beta 0.05 --randomize_normal