#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --nodelist=hkbugpusrv02
#SBATCH --mem=5G
#SBATCH --gres=gpu:1
#SBATCH --error=results/errors/fit_group_ising_model_multi_threshold_aal_%j.err
#SBATCH --output=results/outs/fit_group_ising_model_multi_threshold_aal_%j.out
#SBATCH --job-name="fit_group_ising_model_multi_threshold_aal"

echo ${SLURM_JOB_ID}

srun python fit_group_ising_model_multi_threshold.py --input_directory results/ising_model --output_directory results/ising_model --data_file_name data_ts_all_aal_as_is.pt --num_thresholds 31 --min_threshold 0  --max_threshold 3 --models_per_threshold 101 --max_beta 1.0