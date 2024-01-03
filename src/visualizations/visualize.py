"""
    Loads a pre-trained network
    Extracts some intermediate representation of the data (your training set) from your cnn. This could be the features just before the final classification layer
    Visualize features in a 2D space using t-SNE to do the dimensionality reduction.
    Save the visualization to a file in the reports/figures/ folder.

"""

import os
import sys

import matplotlib.pyplot as plt
import torch
from torch.nn import functional as F

# Add the path to the directory containing the models module to the system path
models_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../models"))
sys.path.append(models_path)

from model import MyAwesomeModel
from sklearn.manifold import TSNE

# load pre-trained model
model = MyAwesomeModel()
model.load_state_dict(torch.load("models/MyAwesomeModel/trained_model.pth"))

# load the data
file_prefix = "data/processed/"
train_images = torch.load(f"{file_prefix}train_images.pt")

# extract features from the model
with torch.no_grad():
    features = model.fc2(F.relu(model.fc1(train_images.view(train_images.shape[0], -1)))).detach().numpy()

# reduce the dimensionality of the features
tsne = TSNE(n_components=2, random_state=0)
features_2d = tsne.fit_transform(features)

train_labels = torch.load(f"{file_prefix}train_labels.pt")

# plot the features
plt.figure(figsize=(6, 5))
plt.scatter(features_2d[:, 0], features_2d[:, 1], c=train_labels.numpy(), cmap="jet")
plt.colorbar()
plt.savefig("reports/figures/tsne.png")
