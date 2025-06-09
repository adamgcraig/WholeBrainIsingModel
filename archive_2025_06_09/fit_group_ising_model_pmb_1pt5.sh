#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --nodelist=hkbugpusrv08
#SBATCH --mem=5G
#SBATCH --gres=gpu:1
#SBATCH --error=results/errors/fit_group_ising_model_mpb_1pt5_%j.err
#SBATCH --output=results/outs/fit_group_ising_model_pmb_1pt5_%j.out
#SBATCH --job-name="fit_group_ising_model_pmb_1pt5"

echo ${SLURM_JOB_ID}

srun python fit_group_ising_model_pmb.py --input_directory results/ising_model --output_directory results/ising_model --threshold_z 1.5 --models_per_subject 5 --max_beta 0.0065 --beta_sim_length 120000 --param_sim_length 1200