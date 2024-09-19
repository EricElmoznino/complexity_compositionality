python train.py --multirun hydra/launcher=mila_eric save_dir=/home/mila/e/eric.elmoznino/scratch/complexity_compositionality/logs experiment=prequential/real_languages \
    hydra.launcher.timeout_min=2880 \
    seed=0,1,2 \
    experiment.data.data_dir=\
/home/mila/e/eric.elmoznino/scratch/complexity_compositionality/data/real_languages/coco-captions/english/,\
/home/mila/e/eric.elmoznino/scratch/complexity_compositionality/data/real_languages/coco-captions/french/,\
/home/mila/e/eric.elmoznino/scratch/complexity_compositionality/data/real_languages/coco-captions/spanish/,\
/home/mila/e/eric.elmoznino/scratch/complexity_compositionality/data/real_languages/coco-captions/german/,\
/home/mila/e/eric.elmoznino/scratch/complexity_compositionality/data/real_languages/coco-captions/japanese/
