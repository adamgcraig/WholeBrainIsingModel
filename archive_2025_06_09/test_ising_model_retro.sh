#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --nodelist=hkbugpusrv07
#SBATCH --mem=5G
#SBATCH --gres=gpu:1
#SBATCH --error=results/errors/test_ising_model_retro_%j.err
#SBATCH --output=results/outs/test_ising_model_retro_%j.out
#SBATCH --job-name="test_ising_model_retro"

echo ${SLURM_JOB_ID}

srun python test_ising_model_retro.py --data_directory data --output_directory results/ising_model --new_sim_length 2000000 --rows_per_file 3249