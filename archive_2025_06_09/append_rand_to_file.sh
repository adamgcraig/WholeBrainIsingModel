#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --nodelist=hkbugpusrv08
#SBATCH --mem=5G
#SBATCH --gres=gpu:1
#SBATCH --error=results/errors/append_rand_to_file_%j.err
#SBATCH --output=results/outs/append_rand_to_file_%j.out
#SBATCH --job-name="append_rand_to_file"

echo ${SLURM_JOB_ID}

FILE_PATTERN=results/ising_model/ising_model_light_group_thresholds_31_min_0_max_3_betas_101_min_1e-10_max_0.05_steps_1200_lr_0.01_beta_updates_8_param_updates_40000_to_thresh_1_reps_1_subj_3348_individual_updates_*000.pt
COUNT=0

while :
do
  for FILE in $FILE_PATTERN; do
    if [ -f $FILE ]; then
      rename 000.pt 000_v$COUNT.pt $FILE --verbose
      ((COUNT++))
    fi
  done
  sleep 5m
done

echo "done"