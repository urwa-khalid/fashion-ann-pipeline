"""Stage 4: evaluate on the test set, write metrics.json and a confusion matrix."""
import json

import matplotlib
matplotlib.use("Agg")  # no display needed
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
from tensorflow import keras

data = np.load("data/processed/data.npz")
x_test, y_test = data["x_test"], data["y_test"]

model = keras.models.load_model("models/model.h5")
loss, accuracy = model.evaluate(x_test, y_test)

with open("metrics.json", "w") as f:
    json.dump({"test_loss": float(loss), "test_accuracy": float(accuracy)}, f, indent=2)

predictions = np.argmax(model.predict(x_test), axis=1)
cm = confusion_matrix(y_test, predictions)
ConfusionMatrixDisplay(cm).plot(cmap="Blues")
plt.title("Fashion-MNIST Confusion Matrix")
plt.savefig("confusion_matrix.png")

print(f"Test accuracy: {accuracy:.4f}")