#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --nodelist=hkbugpusrv06
#SBATCH --mem=5G
#SBATCH --gres=gpu:1
#SBATCH --error=results/errors/find_data_ts_intra_subject_variances_%j.err
#SBATCH --output=results/outs/find_data_ts_intra_subject_variances_%j.out
#SBATCH --job-name="find_data_ts_intra_subject_variances"

echo ${SLURM_JOB_ID}

srun python find_data_ts_intra_subject_variances.py --input_directory results/ising_model --output_directory results/ising_model --ts_file_name binary_data_ts_all.pt --variance_file_name data_fc_intra_subject_variances.pt --range_file_name data_fc_intra_subject_ranges.pt