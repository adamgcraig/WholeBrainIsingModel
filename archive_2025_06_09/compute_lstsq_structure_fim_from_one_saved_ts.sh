#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --nodelist=hkbugpusrv08
#SBATCH --mem=20G
#SBATCH --gres=gpu:1
#SBATCH --error=results/errors/compute_lstsq_structure_fim_from_one_saved_ts_%j.err
#SBATCH --output=results/outs/compute_lstsq_structure_fim_from_one_saved_ts_%j.out
#SBATCH --job-name="compute_lstsq_structure_fim_from_one_saved_ts"

echo ${SLURM_JOB_ID}

srun python compute_lstsq_structure_fim_from_one_saved_ts.py --data_directory results/ising_model --output_directory results/ising_model --device cuda:2 --sim_length 64980 --rep_index 89 --target_index 10