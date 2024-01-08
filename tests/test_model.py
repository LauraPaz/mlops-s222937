from src.models.model import MyAwesomeModel
from src.models.model import MyLightningModel
import pytest
import torch

model = MyAwesomeModel()

def test_my_awesome_model_input_size():
    x = torch.randn(1,28,28)

    assert model(x).shape == torch.Size([1,10])

def test_my_awesome_model_input_size_error():
    with pytest.raises(ValueError, match='Expected input to a 3D tensor'):
        model(torch.randn(1,2))


# def test_my_lightning_model():
#     model = MyLightningModel()
#     x = torch.randn(1,28,28)

#     assert model(x).shape == torch.Size([1,10])
