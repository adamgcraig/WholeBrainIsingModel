#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --nodelist=hkbugpusrv06
#SBATCH --mem=5G
#SBATCH --gres=gpu:1
#SBATCH --error=results/errors/fit_group_ising_model_mpb_aal_0_%j.err
#SBATCH --output=results/outs/fit_group_ising_model_pmb_aal_0_%j.out
#SBATCH --job-name="fit_group_ising_model_pmb_aal_0"

echo ${SLURM_JOB_ID}

srun python fit_group_ising_model_pmb.py --input_directory results/ising_model --output_directory results/ising_model --data_file_name_part all_as_is_aal --threshold_z 0 --models_per_subject 5 --max_beta 1.0 --beta_sim_length 1200 --param_sim_length 1200 --num_saves 1000