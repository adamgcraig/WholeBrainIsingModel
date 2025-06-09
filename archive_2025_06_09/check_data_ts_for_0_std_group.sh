#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --nodelist=hkbugpusrv06
#SBATCH --mem=5G
#SBATCH --gres=gpu:1
#SBATCH --error=results/errors/check_data_ts_for_0_std_group_%j.err
#SBATCH --output=results/outs/check_data_ts_for_0_std_group_%j.out
#SBATCH --job-name="check_data_ts_for_0_std_group"

echo ${SLURM_JOB_ID}

srun python check_data_ts_for_0_std_group.py