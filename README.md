# A Complexity-Based Theory of Compositionality

This code is for reproducing experiments in the paper [*A Complexity-Based Theory of Compositionality*](https://arxiv.org/abs/2410.14817), published at ICML 2025 under the title *Towards a Formal Theory of Representational Compositionality*.

## Dependencies
Install required python packages (should work with newer versions of most packages, but this was the exact environment used):
```
pip install -r requirements.txt
```

We use Weights & Biases for logging experiment data. You can create a free account at [wandb.ai](https://wandb.ai) and run `wandb login` to authenticate your machine.


## Reproducing main experiments

### Generating figures

All paper figures can be reproduced by running the notebook: `papers/theory/notebook.ipynb`.

### Training models

Some cells require you to first run scripts that train models and store metrics in W&B, which the notebook will then pull metrics from using the W&B API. These scripts are located in `scripts/`, and launch training jobs using `hydra` and the `submitit` launcher. The scripts are designed to be run on a computing cluster, but can also be run locally with minor tweaking (see below).

Note that you will need to change minor details in these scripts (and their associated hydra configurations) to run them on your machine:
- Change `logger.entity` in `configs/train.yaml` to your Weights & Biases username.
- Change `save_dir=[your save directory]` in the run scripts to where you want logs and models to be saved.
- If *not* using `submitit` on a computing cluster, remove `--multirun hydra/launcher=[launcher file]` from the run scripts *and* any time a script argument has a "," in it (which sweeps over an argument's value), remove the list of arguments and run the script multiple times with different values for that argument.
- If using `submitit`, you will need to create a new launcher configuration in `configs/hydra/launcher/` and change the `hydra/launcher=[your launcher file]` argument in the run scripts to match your computing cluster's configuration.

### Generating/collecting data to train models

To train models for the emergent language and natural language experiments, you will also need to generate/collect data first.
- For the emergent language experiments, run the `dataloaders/emergent_languages/run.sh` `sbatch` script to generate the data. You'll need to change the `--path` argument in the script to change where the data is saved. If you don't want to use `sbatch` on a cluster but would rather run locally, you'll have to modify the script accordingly.
- For the natural language experiments, you'll first need to collect the COCO captions dataset by running `dataloaders/real_languages/build_dataset.py` (after changing relevant paths inside the script). Then, you'll need to run `dataloaders/real_languages/translate_dataset.py` to get the sentences and representations for each language in the paper (after changing relevant paths inside the script). You'll need to run it separately for each language. Finally, if you want to topological similarities for these languages, you can run `dataloaders/real_languages/topsim.py` (after changing relevant paths inside the script).