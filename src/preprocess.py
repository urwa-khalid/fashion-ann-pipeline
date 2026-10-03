"""Stage 2: normalize pixels to [0, 1] and split a validation set."""

#TESTING

import os

import numpy as np
import yaml
from sklearn.model_selection import train_test_split

with open("params.yaml") as f:
    params = yaml.safe_load(f)["preprocess"]


def normalize(images):
        return (images - images.min()) / (images.max() - images.min())


raw = np.load("data/raw/fashion_mnist.npz")
x_train, y_train = normalize(raw["x_train"]), raw["y_train"]
x_test, y_test = normalize(raw["x_test"]), raw["y_test"]

x_train, x_val, y_train, y_val = train_test_split(
    x_train, y_train,
    test_size=params["val_size"],
    random_state=params["seed"],
)

os.makedirs("data/processed", exist_ok=True)
np.savez("data/processed/data.npz",
         x_train=x_train, y_train=y_train,
         x_val=x_val, y_val=y_val,
         x_test=x_test, y_test=y_test)

print("Train:", x_train.shape, "Val:", x_val.shape, "Test:", x_test.shape)