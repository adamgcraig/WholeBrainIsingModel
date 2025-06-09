#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --nodelist=hkbugpusrv02
#SBATCH --mem=5G
#SBATCH --gres=gpu:1
#SBATCH --error=results/errors/compute_local_individual_correlations_0_short_%j.err
#SBATCH --output=results/outs/compute_local_individual_correlations_0_short_%j.out
#SBATCH --job-name="compute_local_individual_correlations_0_short"

echo ${SLURM_JOB_ID}

srun python compute_local_individual_correlations.py --file_directory results/ising_model --data_file_name_part all_mean_std_0 --model_file_name_part medium_init_uncentered_reps_5_lr_0.01_steps_1200_pupd_per_bopt_1000_num_opt_1_bopt_steps_66_popt_steps_10000 --original_one_over_alpha 20 --num_largest_values 10