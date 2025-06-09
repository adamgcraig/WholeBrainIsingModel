#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --nodelist=hkbugpusrv02
#SBATCH --mem=5G
#SBATCH --gres=gpu:1
#SBATCH --error=results/errors/multi_subject_multi_simulate_beta_test_%j.err
#SBATCH --output=results/outs/multi_subject_multi_simulate_beta_test_%j.out
#SBATCH --job-name="multi_subject_multi_simulate_beta_test"

echo ${SLURM_JOB_ID}

srun python multi_subject_multi_simulate_beta_test.py --data_directory data --output_directory results/ising_model --data_subset training --threshold median --num_parallel 10 --sim_length 12000 --num_updates 100