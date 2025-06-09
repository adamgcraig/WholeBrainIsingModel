#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --nodelist=hkbugpusrv08
#SBATCH --mem=5G
#SBATCH --gres=gpu:1
#SBATCH --error=results/errors/get_data_moments_%j.err
#SBATCH --output=results/outs/get_data_moments_%j.out
#SBATCH --job-name="get_data_moments"

echo ${SLURM_JOB_ID}
srun python get_data_moments.py --input_directory results/ising_model --output_directory results/ising_model