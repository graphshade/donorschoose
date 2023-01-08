#!/bin/bash
#SBATCH --job-name='donorchoose'
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=0-00:10:00
#SBATCH --mem=16gb
#SBATCH --output="result_out_%j.o"
#SBATCH --error="error_out_%j.e"
#SBATCH --mail-user=**********
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --account=******
#SBACTCH --partition=small

module load python3.8.13
python3 ./hpc/tuning.py --filename projects
