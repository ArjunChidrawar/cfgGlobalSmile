#!/bin/bash

#SBATCH --job-name=RunJobs
#SBATCH --partition=sched_mit_sloan_gpu_r8
#SBATCH --ntasks=1
#SBATCH --mem=32GB
#SBATCH --gres=gpu:1
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=aac9@mit.edu
#SBATCH --time=0-23:59:00

# Check GPU status
nvidia-smi

module load sloan/python/3.11.4
module load sloan/python/modules/3.11


cd /Users/arjunchidrawar/Desktop/cfgGlobalSmile/INCLG/

# Path to the Python script
SCRIPT_PATH="/Users/arjunchidrawar/Desktop/cfgGlobalSmile/INCLG/train.py"

# Run the Python script with the configuration file as an argument
python "$SCRIPT_PATH"