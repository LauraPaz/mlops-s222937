from unittest.mock import patch

from hydra import compose, initialize
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, open_dict

from src.train_model import train


@patch('src.models.model.MyAwesomeModel')
def _test_train(mock_model):
    with initialize(config_path="../src/config", version_base="1.1"):
        cfg = compose(config_name="default_config.yaml", return_hydra_config=True)

        # assert cfg == {'training': {'file_prefix': 'data/processed/', 'lr': 0.003, 'epochs': 10}, 'model': {}}
        HydraConfig.instance().set_config(cfg)

        train(cfg)

    mock_model.assert_called_once_with()

# def test_train_fast_dev_run(cfg_train: DictConfig) -> None:
#     """Run for 1 train, val and test step.

#     :param cfg_train: A DictConfig containing a valid training configuration.
#     """
#     HydraConfig().set_config(cfg_train)
#     with open_dict(cfg_train):
#         cfg_train.training.file_prefix = 'data/processed/'
#                 cfg_train.training.file_prefix = 'data/processed/'
#     train(cfg_train)

# @patch('src.models.model.MyAwesomeModel')
# def test_train(mock_model):

#     cfg_train = HydraConfig.get().di
    
#     HydraConfig.instance().set_config(cfg_train)

#     with open_dict(cfg_train):
#         cfg_train.training.file_prefix = 'data/processed/'
#         cfg_train.training.lr = 0.003
#         cfg_train.training.epochs = 10
    
#     train(cfg_train)

#     mock_model.assert_called_once_with()