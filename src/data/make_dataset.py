import torch

if __name__ == "__main__":
    # set the path to the raw data
    file_prefix = "data/raw/"

    # load the data
    train_images = torch.cat([torch.load(f"{file_prefix}train_images_{i}.pt") for i in range(6)], dim=0)
    train_labels = torch.cat([torch.load(f"{file_prefix}train_target_{i}.pt") for i in range(6)], dim=0)
    test_images = torch.load(f"{file_prefix}test_images.pt")
    test_labels = torch.load(f"{file_prefix}test_target.pt")

    # normalize the data
    mean = train_images.mean()
    std = train_images.std()
    train_images = (train_images - mean) / std

    test_images = (test_images - mean) / std

    # save the data
    torch.save(train_images, "data/processed/train_images.pt")
    torch.save(test_images, "data/processed/test_images.pt")
    torch.save(train_labels, "data/processed/train_labels.pt")
    torch.save(test_labels, "data/processed/test_labels.pt")
