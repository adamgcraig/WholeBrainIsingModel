#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --nodelist=hkbugpusrv05
#SBATCH --mem=5G
#SBATCH --gres=gpu:1
#SBATCH --error=results/errors/test_unfitted_ising_model_%j.err
#SBATCH --output=results/outs/test_unfitted_ising_model_%j.out
#SBATCH --job-name="test_unfitted_ising_model"

echo ${SLURM_JOB_ID}

srun python test_unfitted_ising_model.py --data_directory results/ising_model --output_directory results/ising_model --data_file_name_part all_unit_scale --output_file_name_part short --models_per_subject 1 --sim_length 1200 --center_cov --use_inverse_cov --beta 0.0001