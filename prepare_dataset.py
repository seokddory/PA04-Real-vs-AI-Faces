
from pathlib import Path
import random
import shutil

# Original dataset location
source = Path(__file__).resolve().parent / "real_vs_fake" / "real-vs-fake"
# Output directory
output = Path(__file__).resolve().parent

# Fix random seed for reproducibility
random.seed(42)

# Training dataset sizes
model_sizes = [20, 50, 200, 500]

# 1. Create training datasets
for label in ["real", "fake"]:
    source_folder = source / "train" / label

    images = sorted(
        p for p in source_folder.iterdir()
        if p.suffix.lower() in [".jpg", ".jpeg", ".png"]
    )

    # Randomly select 500 images per class
    selected = random.sample(images, 500)

    for size in model_sizes:
        destination = output / f"model_{size}" / label

        # Remove old dataset folder
        if destination.exists():
            shutil.rmtree(destination)

        destination.mkdir(parents=True)

        # Copy selected images
        for image in selected[:size]:
            shutil.copy2(image, destination / image.name)

# 2. Create fixed test dataset
for label in ["real", "fake"]:
    source_folder = source / "test" / label

    images = sorted(
        p for p in source_folder.iterdir()
        if p.suffix.lower() in [".jpg", ".jpeg", ".png"]
    )

    # Select 100 test images per class
    selected = random.sample(images, 100)

    destination = output / "test_original" / label

    if destination.exists():
        shutil.rmtree(destination)

    destination.mkdir(parents=True)

    for image in selected:
        shutil.copy2(image, destination / image.name)

print("Dataset preparation completed!")

for size in model_sizes:
    print(f"Model {size}: Real {size}, Fake {size}")

print("Test: Real 100, Fake 100")
print("Output:", output)
