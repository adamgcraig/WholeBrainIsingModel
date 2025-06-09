#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --nodelist=hkbugpusrv03
#SBATCH --mem=5G
#SBATCH --gres=gpu:1
#SBATCH --error=results/errors/test_ising_model_group_training_%j.err
#SBATCH --output=results/outs/test_ising_model_group_training_%j.out
#SBATCH --job-name="test_ising_model_group_training"

echo ${SLURM_JOB_ID}

srun python test_ising_model_any.py --data_directory data --output_directory results/ising_model --data_subset training --file_suffix group_training_parallel_10000_beta_sims_3_fitting_sims_20_steps_120000_learning_rate_0.1.pt --model_index 5191 --sim_length 120000 --num_sims 1