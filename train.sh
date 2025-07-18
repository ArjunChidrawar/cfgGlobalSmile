#!/bin/bash

#SBATCH -p mit_normal_gpu               # Partition name
#SBATCH -o logs/train_%j.out      # Standard output and error log
#SBATCH -e logs/train_%j.err
#SBATCH --job-name=train
#SBATCH --time=06:00:00             # Max run time
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=20
#SBATCH --mem=200G
#SBATCH --gres=gpu:1                # Request 2 GPUs
#SBATCH --mail-type=END,FAIL        # Email notifications
#SBATCH --mail-user=joyzhuo@mit.edu

module load miniforge/24.3.0-0
conda activate myenv

python train.py
