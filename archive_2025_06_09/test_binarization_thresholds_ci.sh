#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --nodelist=hkbugpusrv01
#SBATCH --mem=5G
#SBATCH --gres=gpu:1
#SBATCH --error=results/errors/test_binarization_thresholds_%j.err
#SBATCH --output=results/outs/test_binarization_thresholds_%j.out
#SBATCH --job-name="test_binarization_thresholds"

echo ${SLURM_JOB_ID}

srun python test_binarization_thresholds_ci.py --input_directory results/ising_model --output_directory results/ising_model --data_subset all --file_name_fragment as_is --num_thresholds 1000