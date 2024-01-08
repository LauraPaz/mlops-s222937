from src.train_model import train
from unittest.mock import patch
from hydra import compose, initialize

@patch('src.models.model.MyAwesomeModel')
def _test_train(mock_model):
    with initialize(config_path="../src/config"):
        cfg = compose(config_name="default_config.yaml")

        assert cfg == {'training': {'file_prefix': 'data/processed/', 'lr': 0.003, 'epochs': 10}, 'model': {}}

        train(cfg)

    mock_model.assert_called_once_with()

