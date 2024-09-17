for language in "fra_Latn" "deu_Latn" "spa_Latn" "zho_Hans" ; do
    sbatch /home/mila/t/thomas.jiralerspong/kolmogorov/real_world_languages/embed_sentences.sh $language
done