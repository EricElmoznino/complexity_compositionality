#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --output=/network/scratch/e/eric.elmoznino/complexity_compositionality/logs/slurm/%x_%j.out

module load miniconda/3
conda activate pytorch

LANGUAGE=$1
python /home/mila/e/eric.elmoznino/complexity_compositionality/dataloaders/real_languages/translate_dataset.py --language $LANGUAGE