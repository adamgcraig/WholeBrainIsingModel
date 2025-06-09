#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --nodelist=hkbugpusrv04
#SBATCH --mem=5G
#SBATCH --gres=gpu:1
#SBATCH --error=results/errors/test_fmri_data_fc_convergence_1_%j.err
#SBATCH --output=results/outs/test_fmri_data_fc_convergence_1_%j.out
#SBATCH --job-name="test_fmri_data_fc_convergence_1"

echo ${SLURM_JOB_ID}

srun python test_fmri_data_fc_convergence.py --data_directory results/ising_model --output_directory results/ising_model --data_file_name_part all_mean_std_1 --min_steps 100 --max_steps 1200 --step_increment 100 --combine_scans --threshold 1 --reverse