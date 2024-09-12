# Real languages prequential with no model parameter reset
python train.py --multirun hydra/launcher=mila_thomas \
    save_dir=/network/scratch/t/thomas.jiralerspong/complexity_compositionality/logs \
    experiment=prequential/real_languages \
    experiment.data.data_dir=/network/scratch/t/thomas.jiralerspong/complexity_compositionality/data/real_languages/coco-captions/ \
    experiment.framework.reset_model_params=False \
    experiment.name="real_languages_prequential_no_reset"


# Real languages full with learn embeddings
python train.py --multirun hydra/launcher=mila_thomas \
    save_dir=/network/scratch/t/thomas.jiralerspong/complexity_compositionality/logs \
    experiment=full_train/real_languages \
    experiment.data.data_dir=/network/scratch/t/thomas.jiralerspong/complexity_compositionality/data/real_languages/coco-captions/ \
    experiment.framework.reset_model_params=True \
    experiment.framework.learn_embeddings=True \
    experiment.name="real_languages_full_learn_embeddings"

# Real languages full with no learn embeddings
python train.py --multirun hydra/launcher=mila_thomas \
    save_dir=/network/scratch/t/thomas.jiralerspong/complexity_compositionality/logs \
    experiment=full_train/real_languages \
    experiment.data.data_dir=/network/scratch/t/thomas.jiralerspong/complexity_compositionality/data/real_languages/coco-captions/ \
    experiment.framework.reset_model_params=True \
    experiment.framework.learn_embeddings=False \
    experiment.name="real_languages_full_no_learn_embeddings"

# Test
# python train.py \
#     save_dir=/network/scratch/t/thomas.jiralerspong/complexity_compositionality/logs \
#     experiment=prequential/real_languages \
#     experiment.data.data_dir=/network/scratch/t/thomas.jiralerspong/complexity_compositionality/data/real_languages/coco-captions/ \
#     experiment.framework.reset_model_params=False \
#     experiment.name="real_languages_prequential_no_reset"

# python train.py \
#     save_dir=/network/scratch/t/thomas.jiralerspong/complexity_compositionality/logs \
#     experiment=full_train/real_languages \
#     experiment.data.data_dir=/network/scratch/t/thomas.jiralerspong/complexity_compositionality/data/real_languages/coco-captions/ \
#     experiment.framework.reset_model_params=True \
#     experiment.name="real_languages_full_reset"

# python train.py --multirun hydra/launcher=mila_thomas save_dir=/network/scratch/t/thomas.jiralerspong/complexity_compositionality/logs \
#     experiment=prequential/real_languages \
#     experiment.data.data_dir=/network/scratch/t/thomas.jiralerspong/complexity_compositionality/data/real_languages/coco-captions/ 

python train.py \
    save_dir=/network/scratch/t/thomas.jiralerspong/complexity_compositionality/logs \
    experiment=prequential/real_languages \
    experiment.data.data_dir=/network/scratch/t/thomas.jiralerspong/complexity_compositionality/data/real_languages/coco-captions/ \
    experiment.framework.reset_model_params=False \
    experiment.name="real_languages_prequential_no_reset"