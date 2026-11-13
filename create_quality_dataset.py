
from pathlib import Path
from PIL import Image

# Project folder
project = Path(__file__).resolve().parent

# Original test dataset
source = project / "test_original"

# Image quality settings
quality_settings = {
    "test_128": 128,
    "test_64": 64,
    "test_32": 32
}

# Create datasets with different image resolutions
for folder_name, resolution in quality_settings.items():

    for label in ["real", "fake"]:
        source_folder = source / label
        output_folder = project / folder_name / label
        output_folder.mkdir(parents=True, exist_ok=True)

        count = 0

        for image_path in sorted(source_folder.iterdir()):
            if image_path.suffix.lower() not in [".jpg", ".jpeg", ".png"]:
                continue

            # Load original image
            image = Image.open(image_path).convert("RGB")
            original_size = image.size

            # Reduce image resolution
            low_res = image.resize(
                (resolution, resolution),
                Image.Resampling.BICUBIC
            )

            # Resize back to original dimensions
            restored = low_res.resize(
                original_size,
                Image.Resampling.BICUBIC
            )

            # Save image as PNG
            output_path = output_folder / (image_path.stem + ".png")
            restored.save(output_path)

            count += 1

        print(f"{folder_name}/{label}: {count} images completed")

print("\nQuality datasets created successfully!")
