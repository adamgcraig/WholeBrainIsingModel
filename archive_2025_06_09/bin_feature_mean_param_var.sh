#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --nodelist=hkbugpusrv03
#SBATCH --mem=5G
#SBATCH --gres=gpu:1
#SBATCH --error=results/errors/bin_feature_mean_param_var_%j.err
#SBATCH --output=results/outs/bin_feature_mean_param_var_%j.out
#SBATCH --job-name="bin_feature_mean_param_var"

echo ${SLURM_JOB_ID}

srun python bin_feature_mean_param_var.py --input_directory results/ising_model --output_directory results/ising_model --file_name_fragment group_training_and_individual_all_signed_params_rectangular_coords_times_beta --num_bins 36 --log10_features