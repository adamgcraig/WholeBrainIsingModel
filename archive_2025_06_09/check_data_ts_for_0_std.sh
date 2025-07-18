#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --nodelist=hkbugpusrv08
#SBATCH --mem=5G
#SBATCH --gres=gpu:1
#SBATCH --error=results/errors/check_data_ts_for_0_std_%j.err
#SBATCH --output=results/outs/check_data_ts_for_0_std_%j.out
#SBATCH --job-name="check_data_ts_for_0_std"

echo ${SLURM_JOB_ID}

srun python check_data_ts_for_0_std.py