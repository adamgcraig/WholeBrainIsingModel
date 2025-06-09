#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --nodelist=hkbugpusrv04
#SBATCH --mem=5G
#SBATCH --gres=gpu:1
#SBATCH --error=results/errors/test_data_ts_flip_rate_%j.err
#SBATCH --output=results/outs/test_data_ts_flip_rate_%j.out
#SBATCH --job-name="test_data_ts_flip_rate"

echo ${SLURM_JOB_ID}
srun python test_data_ts_flip_rate.py --file_directory results/ising_model --ts_file_suffix binary_data_ts_all