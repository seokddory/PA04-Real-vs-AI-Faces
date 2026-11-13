
import os
import numpy as np
from PIL import Image, ImageOps
from tensorflow.keras.models import load_model
from pathlib import Path

# Project folder
project = str(Path(__file__).resolve().parent)

# Model sizes
model_sizes = [20, 50, 200, 500]

# Image size required by Teachable Machine
image_size = (224, 224)

# Fixed test dataset
test_folder = os.path.join(project, "test_original")

for size in model_sizes:
    model_path = os.path.join(
        project, "trained_models", f"model_{size}", "keras_model.h5"
    )

    labels_path = os.path.join(
        project, "trained_models", f"model_{size}", "labels.txt"
    )

    # Load trained model
    model = load_model(model_path, compile=False)

    # Read labels
    with open(labels_path, "r") as f:
        labels = [line.strip().split(" ", 1)[1] for line in f]

    correct = {"real": 0, "fake": 0}
    total = {"real": 0, "fake": 0}

    for true_label in ["real", "fake"]:
        folder = os.path.join(test_folder, true_label)

        for filename in sorted(os.listdir(folder)):
            if not filename.lower().endswith((".jpg", ".jpeg", ".png")):
                continue

            image_path = os.path.join(folder, filename)

            # Preprocess image
            image = Image.open(image_path).convert("RGB")
            image = ImageOps.fit(
                image, image_size, method=Image.Resampling.LANCZOS
            )

            image_array = np.asarray(image, dtype=np.float32)
            image_array = (image_array / 127.5) - 1
            image_array = np.expand_dims(image_array, axis=0)

            # Predict
            prediction = model.predict(image_array, verbose=0)
            predicted_label = labels[np.argmax(prediction)]

            total[true_label] += 1

            if predicted_label.lower() == true_label:
                correct[true_label] += 1

    real_acc = correct["real"] / total["real"] * 100
    fake_acc = correct["fake"] / total["fake"] * 100
    overall_acc = sum(correct.values()) / sum(total.values()) * 100

    print(f"\n===== Model {size} =====")
    print(f"Real Accuracy : {real_acc:.2f}%")
    print(f"Fake Accuracy : {fake_acc:.2f}%")
    print(f"Total Accuracy: {overall_acc:.2f}%")
