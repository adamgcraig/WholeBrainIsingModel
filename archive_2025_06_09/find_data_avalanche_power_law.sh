#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --nodelist=hkbugpusrv02
#SBATCH --mem=5G
#SBATCH --gres=gpu:1
#SBATCH --error=results/errors/find_data_avalanche_power_law_%j.err
#SBATCH --output=results/outs/find_data_avalanche_power_law_%j.out
#SBATCH --job-name="find_data_avalanche_power_law"

echo ${SLURM_JOB_ID}

srun python find_data_avalanche_power_law.py --input_directory results/ising_model --output_directory results/ising_model --data_subset all --file_name_fragment as_is --min_threshold 0 --max_threshold 3 --num_thresholds 3001