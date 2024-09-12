#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=1
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --partition=long,main
#SBATCH --output=/home/mila/t/thomas.jiralerspong/cold_diffusion/scratch/complexity_compositionality/slurm/%x_%j.out

module load anaconda/3
conda activate kolmogorov
python /home/mila/t/thomas.jiralerspong/kolmogorov/complexity_compositionality/dataloaders/real_languages/build_dataset.py