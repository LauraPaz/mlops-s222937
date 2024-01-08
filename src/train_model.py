import logging
import os

import hydra
import matplotlib.pyplot as plt
import torch
from rich.logging import RichHandler
from torch import nn, optim

import wandb
from src.data.make_dataset import mnist
from src.models.model import MyAwesomeModel


@hydra.main(config_path="config", config_name="default_config.yaml")
def train(config):

    wandb.init()

    log = get_logger()
    cfg = config.training

    model = MyAwesomeModel()

    # Magic
    wandb.watch(model, log_freq=100)

    train_set, _ = mnist()

    train_loader = torch.utils.data.DataLoader(train_set, batch_size=64, shuffle=True)

    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.parameters(), lr=cfg.lr)

    losses = []
    for e in range(cfg.epochs):
        running_loss = 0
        for images, labels in train_loader:
            optimizer.zero_grad()
            log_ps = model(images)
            loss = criterion(log_ps, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        else:
            training_loss = running_loss / len(train_loader)

            log.info(f"Training loss: {training_loss}")
            wandb.log({"Training loss": training_loss})
            losses.append(training_loss)

            # plot a calendar heatmap of the weights of the last layer
            plt.imshow(model.fc3.weight.detach().numpy())
            # log the plot to wandb
            wandb.log({"fc3 weights heatmap": plt})

            # plot a histogram of the weights of the last layer
            plt.hist(model.fc3.weight.detach().numpy().flatten(), bins=10)
            # log the plot to wandb
            wandb.log({"fc3 weights histogram": wandb.Image(plt)})

    # create a folder for the model with the class name of the model
    os.makedirs(f"models/{model.__class__.__name__}", exist_ok=True)
    # Save the model
    torch.save(model.state_dict(), f"models/{model.__class__.__name__}/trained_model.pth")
    log.info(f"Model trained and saved at models/{model.__class__.__name__}/trained_model.pth")

    # make a plot of the training loss
    plt.plot(losses)
    plt.savefig("reports/figures/training_loss.png")


def get_logger():
    hydra_path = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
    job_name = hydra.core.hydra_config.HydraConfig.get().job.name

    file_handler = logging.FileHandler(os.path.join(hydra_path, f"{job_name}.log"))
    rich_handler = RichHandler()

    log = logging.getLogger(__name__)
    log.setLevel(logging.DEBUG)
    log.addHandler(rich_handler)
    log.addHandler(file_handler)
    log.propagate = False
    log.info("Successfully create rich logger")

    return log


if __name__ == "__main__":
    train()
