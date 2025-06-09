#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --nodelist=hkbugpusrv06
#SBATCH --mem=5G
#SBATCH --gres=gpu:1
#SBATCH --error=results/errors/test_ising_model_flip_rate_vs_beta_short_%j.err
#SBATCH --output=results/outs/test_ising_model_flip_rate_vs_beta_short_%j.out
#SBATCH --job-name="test_ising_model_flip_rate_vs_beta_short"

echo ${SLURM_JOB_ID}
srun python test_ising_model_flip_rate_vs_beta.py --file_directory results/ising_model --model_file_suffix group_training_and_individual_all_fold_1_betas_5_steps_1200_lr_0.01 --sim_length 1200 --zero_h --num_beta 10