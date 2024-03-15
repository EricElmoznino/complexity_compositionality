import hydra
from utils import ast_eval
from lightning import Trainer, seed_everything
import warnings

warnings.filterwarnings("ignore", ".*does not have many workers.*")

import wandb


@hydra.main(config_path="./configs/", config_name="train", version_base=None)
def train(cfg):
    seed_everything(cfg.seed)
    datamodule = hydra.utils.instantiate(cfg.experiment.data)
    framework = hydra.utils.instantiate(cfg.experiment.framework)
    logger = hydra.utils.instantiate(cfg.logger) if cfg.logger else False
    callbacks = (
        hydra.utils.instantiate(cfg.experiment.callbacks)
        if cfg.experiment.callbacks
        else None
    )

    trainer = Trainer(
        logger=logger,
        callbacks=callbacks,
        enable_checkpointing=False,
        **cfg.experiment.trainer,
    )
    trainer.fit(model=framework, datamodule=datamodule)


if __name__ == "__main__":
    """Run with:
    `python train.py experiment=[experiment_folder]/[experiment_name].yaml [overrides]`
    """
    train()
