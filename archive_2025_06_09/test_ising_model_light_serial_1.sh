#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --nodelist=hkbugpusrv02
#SBATCH --mem=5G
#SBATCH --gres=gpu:1
#SBATCH --error=results/errors/test_ising_model_light_serial_1_%j.err
#SBATCH --output=results/outs/test_ising_model_light_serial_1_%j.out
#SBATCH --job-name="test_ising_model_light_serial_1"

echo ${SLURM_JOB_ID}

srun python test_ising_model_light_series.py --data_directory results/ising_model --output_directory results/ising_model  --data_file_name_part all_mean_std_1 --model_file_fragment ising_model_light_all_mean_std_1_medium_init_uncentered_reps_5_lr_0.01_steps_1200_pupd_per_bopt_1000_num_opt_1_bopt_steps_68_popt_steps --min_updates 46000 --max_updates 53000