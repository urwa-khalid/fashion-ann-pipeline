"""Stage 3: build and train the ANN, save the model and training history."""

import os

import numpy as np
import pandas as pd
import yaml
from tensorflow import keras

with open("params.yaml") as f:
    params = yaml.safe_load(f)["train"]

data = np.load("data/processed/data.npz")

model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),
    keras.layers.Dense(params["dense_units"], activation="relu"),
    keras.layers.Dropout(params["dropout_rate"]),
    keras.layers.Dense(10, activation="softmax"),
])

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=params["learning_rate"]),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)

history = model.fit(
    data["x_train"], data["y_train"],
    validation_data=(data["x_val"], data["y_val"]),
    epochs=params["epochs"],
    batch_size=params["batch_size"],
)

os.makedirs("models", exist_ok=True)
model.save("models/model.h5")
pd.DataFrame(history.history).to_csv("models/history.csv", index=False)

print("Model and history saved in models/")