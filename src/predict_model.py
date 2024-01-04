import click
import numpy as np
import torch

from models.model import MyAwesomeModel


@click.command()
@click.argument("model_path", type=click.Path(exists=True, file_okay=True, dir_okay=False))
@click.argument("data_path", type=click.Path(exists=True, readable=True))
def predict(model_path, data_path):
    """Run prediction for a given model and dataloader.

    Args:
        model: model to use for prediction
        dataloader: dataloader with batches

    Returns
        Tensor of shape [N, d] where N is the number of samples and d is the output dimension of the model

    """

    model = MyAwesomeModel()
    model.load_state_dict(torch.load(model_path))
    model.eval()

    images = load_and_preprocess_images(data_path)

    data_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(images), batch_size=64, shuffle=False)

    print(f"Number of samples: {len(data_loader.dataset)}")
    print(f"Output dimension of the model: {model.state_dict()['fc3.weight'].shape[0]}")

    # print the type and shape of the 1st element in the data_loader
    print(f"Type of the 1st element in the data_loader: {type(next(iter(data_loader))[0])}")

    with torch.no_grad():
        all_predictions = []
        for batch in data_loader:
            # If your model expects a single image, iterate over images in the batch
            predictions = torch.cat([torch.exp(model(image)) for image in batch], 0)
            all_predictions.append(predictions)

        # Concatenate predictions from all batches
        ps = torch.cat(all_predictions, 0)
        top_p, top_class = ps.topk(1, dim=1)

        # Make sure the shape of the output is [N, d]
        print(f"Shape of the output: {top_class.shape}")

    return top_p, top_class


def load_and_preprocess_images(data_path: str) -> torch.Tensor:
    """Load and preprocess images from a given path.

    Args:
        data_path: path to the data

    """

    if data_path.endswith(".npy"):
        # Load images from a numpy file
        images = np.load(data_path)
        images = torch.from_numpy(images).float()
    elif data_path.endswith(".pt"):
        images = torch.load(data_path)
    else:
        raise ValueError(f"Unknown file format for {data_path}")

    # normalize the data
    mean = images.mean()
    std = images.std()
    images = (images - mean) / std

    # print the object type and shape
    print(f"Object type: {type(images)}")
    print(f"Object shape: {images.shape}")

    return images


if __name__ == "__main__":
    predict()
    # 'models/MyAwesomeModel/trained_model.pth', 'data/processed/test_images.pt'
