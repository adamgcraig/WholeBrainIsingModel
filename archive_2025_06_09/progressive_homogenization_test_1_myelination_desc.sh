#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --nodelist=hkbugpusrv07
#SBATCH --mem=5G
#SBATCH --gres=gpu:1
#SBATCH --error=results/errors/homogenize_single_param_test_1_myelination_desc_%j.err
#SBATCH --output=results/outs/homogenize_single_param_test_1_myelination_desc_%j.out
#SBATCH --job-name="progressive_homogenization_test_1_myelination_desc"

echo ${SLURM_JOB_ID}

srun python progressive_homogenization_test.py --data_directory results/ising_model --output_directory results/ising_model  --combine_scans --homogenize_h --use_abs --descend