#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --nodelist=hkbugpusrv02
#SBATCH --mem=5G
#SBATCH --gres=gpu:1
#SBATCH --error=results/errors/data_ts_bin_to_pt_%j.err
#SBATCH --output=results/outs/data_ts_bin_to_pt_%j.out
#SBATCH --job-name="data_ts_bin_to_pt"

echo ${SLURM_JOB_ID}

srun python data_ts_bin_to_pt.py --data_directory data --output_directory data