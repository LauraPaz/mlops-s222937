import os

import matplotlib.pyplot as plt
import torch
from torch import nn, optim
import hydra
import wandb
from data.make_dataset import mnist
from models.model import MyLightningModel
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning import Trainer
from pytorch_lightning import loggers

@hydra.main(config_path="config", config_name="default_config.yaml")
def train(config):
    cfg = config.training

    model = MyLightningModel()

    checkpoint_callback = ModelCheckpoint(
        dirpath="./models", monitor="val_loss", mode="min"
    )

    train_set, _ = mnist()

    train_loader = torch.utils.data.DataLoader(train_set, batch_size=64, shuffle=True)

    trainer = Trainer(
        callbacks=[checkpoint_callback],
        max_epochs=cfg.epochs,
        limit_train_batches=0.2, 
        logger=loggers.WandbLogger(project="mlops-s222937"),
        precision=16,
        profiler='simple',
    )
    trainer.fit(model, train_loader)

if __name__ == "__main__":
    train()
