#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --nodelist=hkbugpusrv08
#SBATCH --mem=5G
#SBATCH --gres=gpu:1
#SBATCH --error=results/errors/beta_test_ising_model_pmb_pmb_1_%j.err
#SBATCH --output=results/outs/beta_test_ising_model_pmb_1_%j.out
#SBATCH --job-name="beta_test_ising_model_pmb_1"

echo ${SLURM_JOB_ID}

srun python beta_test_ising_model_pmb.py --data_directory results/ising_model --output_directory results/ising_model --data_file_name_part group_threshold_1 --model_file_fragment pseudolikelihood_mean_std_1_models_5_rand_min_-100_max_100_lr_0.001_steps_1000000_num_beta_837_min_1e-10_max_1_updates_1000_sim_120000 --sim_length 120000 --max_beta 1.0