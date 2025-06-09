#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --nodelist=hkbugpusrv04
#SBATCH --mem=5G
#SBATCH --gres=gpu:1
#SBATCH --error=results/errors/test_ising_model_fc_convergence_1_loop_%j.err
#SBATCH --output=results/outs/test_ising_model_fc_convergence_1_loop_%j.out
#SBATCH --job-name="test_ising_model_fc_convergence_1_loop"

echo ${SLURM_JOB_ID}
for i in {1..1000000}
do
 srun python test_ising_model_fc_convergence.py --data_directory results/ising_model --output_directory results/ising_model --data_file_name_part all_mean_std_1 --model_file_fragment all_mean_std_1_medium_init_uncentered_reps_5_lr_0.01_steps_1200_pupd_per_bopt_1000_num_opt_1_bopt_steps_68_popt_steps_40000 --min_steps 100 --max_steps 120000 --step_increment 100 --combine_scans
done