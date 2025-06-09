#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --nodelist=hkbugpusrv02
#SBATCH --mem=5G
#SBATCH --gres=gpu:1
#SBATCH --error=results/errors/make_selected_avalanche_hist_counts_%j.err
#SBATCH --output=results/outs/make_selected_avalanche_hist_counts_%j.out
#SBATCH --job-name="make_selected_avalanche_hist_counts"

echo ${SLURM_JOB_ID}

srun python make_selected_avalanche_hist_counts.py --input_directory results/ising_model --output_directory results/ising_model --data_subset all --file_name_fragment as_is