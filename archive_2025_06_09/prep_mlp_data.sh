#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --nodelist=hkbugpusrv03
#SBATCH --mem=5G
#SBATCH --gres=gpu:1
#SBATCH --error=results/errors/prep_mlp_data_%j.err
#SBATCH --output=results/outs/prep_mlp_data_%j.out
#SBATCH --job-name="prep_mlp_data"

echo ${SLURM_JOB_ID}

srun python prep_mlp_data.py --subject_list_directory data --input_directory results/ising_model --output_directory data/ml_examples --feature_file_name_fragment group_training_and_individual_all --model_file_name_fragment group_training_and_individual_all_fold_1_betas_5_steps_1200_lr_0.01