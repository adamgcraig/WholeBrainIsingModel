#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --nodelist=hkbugpusrv03
#SBATCH --mem=5G
#SBATCH --gres=gpu:1
#SBATCH --error=results/errors/fit_ising_models_one_subject_%j.err
#SBATCH --output=results/outs/fit_ising_models_one_subject_%j.out
#SBATCH --job-name="fit_ising_models_one_subject"

echo ${SLURM_JOB_ID}

srun python fit_ising_models_one_subject.py --data_directory data --output_directory results/ising_model --subject_id 516742 --num_epochs 100000 --save_every_epochs 500 --print_every_epochs 20 --num_nodes 360 --window_length 1200
