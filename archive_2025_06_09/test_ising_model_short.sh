#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --nodelist=hkbugpusrv01
#SBATCH --mem=5G
#SBATCH --gres=gpu:1
#SBATCH --error=results/errors/test_ising_model_short_%j.err
#SBATCH --output=results/outs/test_ising_model_short_%j.out
#SBATCH --job-name="test_ising_model_short"

echo ${SLURM_JOB_ID}

srun python test_ising_model_short.py --data_directory results/ising_model --model_directory results/ising_model --result_directory results/ising_model --mean_state_file_name data_mean_state_aal_short_threshold_1.pt --mean_state_product_file_name data_mean_state_product_aal_short_threshold_1.pt --model_file_name ising_model_aal_short_threshold_1_beta_updates_9_param_updates_33000.pt --sim_length 120000