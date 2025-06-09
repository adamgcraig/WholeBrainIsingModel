#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --nodelist=hkbugpusrv01
#SBATCH --mem=5G
#SBATCH --gres=gpu:1
#SBATCH --error=results/errors/find_fc_vs_beta_%j.err
#SBATCH --output=results/outs/find_fc_vs_beta_%j.out
#SBATCH --job-name="find_fc_vs_beta"

echo ${SLURM_JOB_ID}

srun python find_fc_vs_beta.py --data_directory results/ising_model --output_directory results/ising_model --data_file_name_part all_unit_scale --output_file_name_part short --models_per_subject 1000 --sim_length 1200 --num_targets 1 --use_inverse_cov --max_beta 1