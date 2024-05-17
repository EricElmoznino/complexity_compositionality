BASE_DIR=/home/mila/e/eric.elmoznino/scratch/complexity_compositionality/data
BASE_NAME=emergent_languages_2attr_8vals

python train.py --multirun hydra/launcher=mila_eric save_dir=/home/mila/e/eric.elmoznino/scratch/complexity_compositionality/logs \
experiment=prequential/emergent_languages \
++experiment.data.data_dir=\
"${BASE_DIR}/${BASE_NAME}_resets_seed-1",\
"${BASE_DIR}/${BASE_NAME}_no-resets_seed-1",\
"${BASE_DIR}/${BASE_NAME}_resets_seed-2",\
"${BASE_DIR}/${BASE_NAME}_no-resets_seed-2",\
"${BASE_DIR}/${BASE_NAME}_resets_seed-3",\
"${BASE_DIR}/${BASE_NAME}_no-resets_seed-3",\
"${BASE_DIR}/${BASE_NAME}_resets_seed-4",\
"${BASE_DIR}/${BASE_NAME}_no-resets_seed-4",\
"${BASE_DIR}/${BASE_NAME}_resets_seed-5",\
"${BASE_DIR}/${BASE_NAME}_no-resets_seed-5"