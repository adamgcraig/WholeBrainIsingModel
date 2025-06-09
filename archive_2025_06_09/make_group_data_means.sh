#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --nodelist=hkbugpusrv03
#SBATCH --mem=5G
#SBATCH --gres=gpu:1
#SBATCH --error=results/errors/make_group_data_means_1_%j.err
#SBATCH --output=results/outs/make_group_data_means_1_%j.out
#SBATCH --job-name="make_group_data_means_1"

echo ${SLURM_JOB_ID}

srun python make_group_data_means.py --input_directory results/ising_model --output_directory results/ising_model --threshold_z 1