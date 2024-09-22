#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --mem=8G
#SBATCH --time=24:00:00
#SBATCH --output=/network/scratch/e/eric.elmoznino/complexity_compositionality/logs/slurm/%x_%j.out

module load miniconda/3
conda activate pytorch

python /home/mila/e/eric.elmoznino/complexity_compositionality/dataloaders/real_languages/topsim.py