#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --mem=8G
#SBATCH --time=24:00:00
#SBATCH --output=/network/scratch/t/thomas.jiralerspong/complexity_compositionality/logs/slurm/%x_%j.out

module load miniconda/3
conda activate kolmogorov

LANGUAGE=$1
python /home/mila/t/thomas.jiralerspong/kolmogorov/complexity_compositionality/data/translate_sentences.py --language $LANGUAGE --model_name facebook/nllb-200-distilled-600M --batch_size 100