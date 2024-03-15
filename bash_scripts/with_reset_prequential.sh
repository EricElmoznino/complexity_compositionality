#!/bin/bash

#SBATCH --partition=long,main,unkillable,lab-bengioy
#SBATCH --gres=gpu:1                                    
#SBATCH --mem=8G                                       
#SBATCH --time=8:00:00                                
#SBATCH -o /network/scratch/t/thomas.jiralerspong/kolmogorov/slurm/slurm-%j.out

# module load miniconda/3
conda activate kolmogorov
cd /home/mila/t/thomas.jiralerspong/kolmogorov/complexity_compositionality

python train.py experiment=prequential/emergent_languages_with_reset