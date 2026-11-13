
import os
import numpy as np
from PIL import Image, ImageOps
from tensorflow.keras.models import load_model
from pathlib import Path
# Project directory
project = str(Path(__file__).resolve().parent)

# Load trained Model 500
model_path = os.path.join(
    project, "trained_models", "model_500", "keras_model.h5"
)

labels_path = os.path.join(
    project, "trained_models", "model_500", "labels.txt"
)

model = load_model(model_path, compile=False)

# Read class labels
with open(labels_path, "r") as f:
    labels = [
        line.strip().split(" ", 1)[1].lower()
        for line in f
    ]

# Test datasets
test_sets = {
    "Original": "test_original",
    "128x128": "test_128",
    "64x64": "test_64",
    "32x32": "test_32"
}

image_size = (224, 224)

# Evaluate each image quality condition
for condition, folder_name in test_sets.items():

    correct = {"real": 0, "fake": 0}
    total = {"real": 0, "fake": 0}

    for true_label in ["real", "fake"]:
        folder = os.path.join(project, folder_name, true_label)

        for filename in sorted(os.listdir(folder)):
            if not filename.lower().endswith((".jpg", ".jpeg", ".png")):
                continue

            image_path = os.path.join(folder, filename)

            # Load and preprocess image
            image = Image.open(image_path).convert("RGB")
            image = ImageOps.fit(
                image,
                image_size,
                method=Image.Resampling.LANCZOS
            )

            image_array = np.asarray(image, dtype=np.float32)
            image_array = (image_array / 127.5) - 1
            image_array = np.expand_dims(image_array, axis=0)

            # Model prediction
            prediction = model.predict(image_array, verbose=0)
            predicted_label = labels[np.argmax(prediction)]

            total[true_label] += 1

            if predicted_label == true_label:
                correct[true_label] += 1

    # Calculate accuracy
    real_acc = correct["real"] / total["real"] * 100
    fake_acc = correct["fake"] / total["fake"] * 100
    overall_acc = sum(correct.values()) / sum(total.values()) * 100

    # Print results
    print(f"\n===== {condition} =====")
    print(f"Real Accuracy : {real_acc:.2f}%")
    print(f"Fake Accuracy : {fake_acc:.2f}%")
    print(f"Total Accuracy: {overall_acc:.2f}%")
    print(f"Correct       : {sum(correct.values())}/{sum(total.values())}")
