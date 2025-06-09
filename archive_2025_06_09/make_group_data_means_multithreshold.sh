#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --nodelist=hkbugpusrv06
#SBATCH --mem=5G
#SBATCH --gres=gpu:1
#SBATCH --error=results/errors/make_group_data_means_multithreshold_%j.err
#SBATCH --output=results/outs/make_group_data_means_multithreshold_%j.out
#SBATCH --job-name="make_group_data_means_multithreshold"

echo ${SLURM_JOB_ID}

srun python make_group_data_means_multithreshold.py --input_directory results/ising_model --output_directory results/ising_model --data_ts_string all_as_is --num_thresholds 31 --min_threshold 0  --max_threshold 3