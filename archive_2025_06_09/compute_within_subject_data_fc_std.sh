#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --nodelist=hkbugpusrv06
#SBATCH --mem=5G
#SBATCH --gres=gpu:1
#SBATCH --error=results/errors/compute_within_subject_data_fc_std_1_%j.err
#SBATCH --output=results/outs/compute_within_subject_data_fc_std_1_%j.out
#SBATCH --job-name="compute_within_subject_data_fc_std_1"

echo ${SLURM_JOB_ID}

srun python compute_within_subject_data_fc_std.py --data_directory results/ising_model --output_directory results/ising_model  --data_file_name_part all_as_is --window_increment 100 --min_window 200 --max_window 1200 --threshold 1.0