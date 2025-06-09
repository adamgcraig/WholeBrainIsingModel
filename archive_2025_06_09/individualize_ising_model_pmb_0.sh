#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --nodelist=hkbugpusrv07
#SBATCH --mem=5G
#SBATCH --gres=gpu:1
#SBATCH --error=results/errors/individualize_ising_model_mpb_0_%j.err
#SBATCH --output=results/outs/individualize_ising_model_pmb_0_%j.out
#SBATCH --job-name="individualize_ising_model_pmb_0"

echo ${SLURM_JOB_ID}

srun python individualize_ising_model_pmb.py --input_directory results/ising_model --output_directory results/ising_model --group_model_file_fragment light_group_threshold_0_betas_5_min_1e-10_max_0.01_beta_steps_1200_param_steps_1200_lr_0.01_beta_updates_71 --threshold_z 0