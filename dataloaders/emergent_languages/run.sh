#!/usr/bin/env bash
#SBATCH --gres=gpu:1
#SBATCH --time=23:00:00
#SBATCH --mem=16G

module load miniconda/3
conda activate pytorch

for seed in 1 2 3 4 5
do
    python train.py --path "/home/mila/e/eric.elmoznino/scratch/complexity_compositionality/data/emergent_languages_2attr_8vals_resets_seed-$seed" --n_object_save_repeats 50 --seed $seed --valid_num 0

    python train.py --path "/home/mila/e/eric.elmoznino/scratch/complexity_compositionality/data/emergent_languages_2attr_8vals_no-resets_seed-$seed" --noresets --n_object_save_repeats 50 --seed $seed --valid_num 0
done

