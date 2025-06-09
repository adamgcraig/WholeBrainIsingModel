#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --nodelist=hkbugpusrv08
#SBATCH --mem=5G
#SBATCH --gres=gpu:1
#SBATCH --error=results/errors/individualize_ising_model_mpb_1_%j.err
#SBATCH --output=results/outs/individualize_ising_model_pmb_1_%j.out
#SBATCH --job-name="individualize_ising_model_pmb_1"

echo ${SLURM_JOB_ID}

srun python individualize_ising_model_pmb.py --input_directory results/ising_model --output_directory results/ising_model --group_model_file_fragment light_group_threshold_1_betas_5_min_1e-10_max_0.01_beta_steps_1200_param_steps_1200_lr_0.01_beta_updates_66 --threshold_z 1