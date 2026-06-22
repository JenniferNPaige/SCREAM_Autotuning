#!/bin/bash
#SBATCH -N 1
#SBATCH -C cpu
#SBATCH -q regular
#SBATCH -J optimizing
#SBATCH --mail-user=jpaige@ucdavis.edu
#SBATCH --mail-type=ALL
#SBATCH --array=10-20
#SBATCH --cpus-per-task=1
#SBATCH --time=02:10:00

module load conda
conda activate ESEm_copy

#run the application:
#python /global/cfs/cdirs/e3sm/jpaige3/optimizing/run_GPsurrogate_fromsave_final_efficient.py
python /global/cfs/cdirs/e3sm/jpaige3/optimizing/run_GPsurrogate_fromsave_final_efficient.py --seed $SLURM_ARRAY_TASK_ID --nstarts 10

# this is from https://my.nersc.gov/script_generator.php