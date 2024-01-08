import torch
import torch.nn.functional as F
import wandb
from pytorch_lightning import LightningModule

class MyNeuralNet(torch.nn.Module):
    """Basic neural network class.

    Args:
        in_features: number of input features
        out_features: number of output features

    """

    def __init__(self, in_features: int, out_features: int) -> None:
        self.l1 = torch.nn.Linear(in_features, 500)
        self.l2 = torch.nn.Linear(500, out_features)
        self.r = torch.nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the model.

        Args:
            x: input tensor expected to be of shape [N,in_features]

        Returns:
            Output tensor with shape [N,out_features]

        """
        return self.l2(self.r(self.l1(x)))


class MyAwesomeModel(torch.nn.Module):
    """My awesome model."""

    def __init__(self):
        super().__init__()
        self.fc1 = torch.nn.Linear(784, 128)
        self.fc2 = torch.nn.Linear(128, 64)
        self.fc3 = torch.nn.Linear(64, 10)

    def forward(self, x):
        if x.ndim != 3:
            raise ValueError('Expected input to a 3D tensor')
        if x.shape[0] != 1 or x.shape[1] != 28 or x.shape[2] != 28:
            raise ValueError('Expected each sample to have shape [1, 28, 28]')

        # make sure input tensor is flattened
        x = x.view(x.shape[0], -1)

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.log_softmax(self.fc3(x), dim=1)

        return x
    

class MyLightningModel(LightningModule):
    """My awesome lightning model."""

    def __init__(self):
        super().__init__()
        self.backbone = torch.nn.Sequential(
            torch.nn.Conv2d(1, 32, 3),  # [B, 1, 28, 28] -> [B, 32, 26, 26]
            torch.nn.LeakyReLU(),
            torch.nn.Conv2d(32, 64, 3), # [B, 32, 26, 26] -> [B, 64, 24, 24]
            torch.nn.LeakyReLU(),
            torch.nn.MaxPool2d(2),      # [B, 64, 24, 24] -> [B, 64, 12, 12]
            torch.nn.Flatten(),        # [B, 64, 12, 12] -> [B, 64 * 12 * 12]
            torch.nn.Linear(144, 10),
            
        )

        self.criterion = torch.nn.CrossEntropyLoss()

    def forward(self, x):
        return self.backbone(x)
    
    def training_step(self, batch, batch_idx):
        data, target = batch
        preds = self(data)

         # self.logger.experiment is the same as wandb.log
        self.logger.experiment.log({'logits': wandb.Histogram(preds.detach().numpy())})

        loss = self.criterion(preds, target)
        acc = (target == preds.argmax(dim=-1)).float().mean()
        self.log('train_loss', loss)
        self.log('train_acc', acc)
        return loss
    
    def validation_step(self, batch, batch_idx):
        data, target = batch
        preds = self(data)
        loss = self.criterion(preds, target)
        acc = (target == preds.argmax(dim=-1)).float().mean()
        self.log('val_loss', loss)
        self.log('val_acc', acc)
        return loss
    
    def test_step(self, batch, batch_idx):
        data, target = batch
        preds = self(data)
        loss = self.criterion(preds, target)
        acc = (target == preds.argmax(dim=-1)).float().mean()
        self.log('test_loss', loss)
        self.log('test_acc', acc)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-2)