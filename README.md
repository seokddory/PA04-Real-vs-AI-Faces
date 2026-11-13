
# Real vs. AI-Generated Face Classification

## 1. Project Overview

This project investigates the classification of real human faces and AI-generated faces using image classification models trained with Google Teachable Machine.

The project focuses on two research questions:

- **RQ1 — Training Dataset Size:** How does the number of training images affect classification accuracy for real and AI-generated faces?
- **RQ2 — Image Quality:** How does image resolution affect classification accuracy for real and AI-generated faces?

RQ1 compares four models trained with different numbers of images. RQ2 evaluates the effect of reducing test image resolution using the model trained with the largest dataset.

## 2. Dataset

**Dataset Source:** [140K Real and Fake Faces — Kaggle](https://www.kaggle.com/datasets/xhlulu/140k-real-and-fake-faces)

The dataset contains real human face images from FFHQ and AI-generated face images produced using StyleGAN.

The original dataset provides separate training, validation, and test splits.

For this project, images were sampled from the original training and test splits using a fixed random seed (`42`). The original validation split was not used in the custom evaluation.

### 2.1 Training Dataset

Four training datasets were prepared for RQ1.

| Model | Real Images | Fake Images | Total Images |
|---|---:|---:|---:|
| Model 20 | 20 | 20 | 40 |
| Model 50 | 50 | 50 | 100 |
| Model 200 | 200 | 200 | 400 |
| Model 500 | 500 | 500 | 1000 |

The training subsets are nested, meaning that the images in smaller training sets are included in larger training sets.

### 2.2 Test Dataset

All models were evaluated using the same fixed test dataset.

| Class | Number of Images |
|---|---:|
| Real | 100 |
| Fake | 100 |
| Total | 200 |

The test images were selected from the original test split and were not included in the training subsets.

## 3. Environment Setup

The local experiments were evaluated on macOS using Python 3.10.

The main software versions were:

| Component | Version |
|---|---|
| Operating System | macOS |
| Python | 3.10 |
| TensorFlow | 2.15.0 |
| NumPy | 1.26.4 |
| Pillow | 12.3.0 |

The environment configuration is provided in `environment.yml`.

To create and activate the Conda environment, run:

```bash
conda env create -f environment.yml
conda activate ai_lab2
```

The environment file records the main dependencies used in the experiments. Installation may require adjustments depending on the operating system and hardware.

## 4. Dataset Preparation

### Step 1. Download the Original Dataset

Download the [140K Real and Fake Faces dataset](https://www.kaggle.com/datasets/xhlulu/140k-real-and-fake-faces) from Kaggle.

Extract the dataset and place the original `train`, `valid`, and `test` directories in the following location:

```text
PA04_Project/
└── real_vs_fake/
    └── real-vs-fake/
        ├── train/
        │   ├── real/
        │   └── fake/
        ├── valid/
        │   ├── real/
        │   └── fake/
        └── test/
            ├── real/
            └── fake/
```

The `real_vs_fake/` directory is excluded from Git tracking because the original dataset is provided separately through Kaggle.

### Step 2. Generate Training and Test Subsets

Run the dataset preparation script from the project directory:

```bash
python prepare_dataset.py
```

The script uses a fixed random seed (`42`) and sorted filenames to select the images.

It creates the following directories:

```text
model_20/
model_50/
model_200/
model_500/
test_original/
```

Each directory contains separate `real/` and `fake/` subdirectories.

The smaller training subsets are contained in the larger training subsets, while all four models share the same fixed test dataset.

**Note:** The preparation script recreates its output directories. Existing generated dataset folders should be backed up before rerunning it.

## 5. Model Training

The models were trained using [Google Teachable Machine](https://teachablemachine.withgoogle.com/).

### Step 1. Create an Image Classification Project

Open Google Teachable Machine and select:

**Get Started → Image Project → Standard Image Model**

Create two classes:

- Class 1: `real`
- Class 2: `fake`

### Step 2. Upload Training Images

Create and train four separate models using the prepared datasets.

| Model | Real Images | Fake Images | Dataset Directory |
|---|---:|---:|---|
| Model 20 | 20 | 20 | `model_20/` |
| Model 50 | 50 | 50 | `model_50/` |
| Model 200 | 200 | 200 | `model_200/` |
| Model 500 | 500 | 500 | `model_500/` |

For each model, upload images from the corresponding `real/` directory to the `real` class and images from the `fake/` directory to the `fake` class.

### Step 3. Configure Training Parameters

The same training settings were used for all four models.

| Parameter | Value |
|---|---|
| Model Type | Standard Image Model |
| Epochs | 50 |
| Batch Size | 16 |
| Learning Rate | 0.001 |

Only the number of training images was changed between the four RQ1 conditions.

### Step 4. Train and Export the Models

Train each model separately using the specified settings.

After training, select:

**Export Model → TensorFlow → Keras → Download**

Extract the downloaded files and save them in the following structure:

```text
trained_models/
├── model_20/
│   ├── keras_model.h5
│   └── labels.txt
├── model_50/
│   ├── keras_model.h5
│   └── labels.txt
├── model_200/
│   ├── keras_model.h5
│   └── labels.txt
└── model_500/
    ├── keras_model.h5
    └── labels.txt
```

The trained Keras model files and their corresponding class labels are included in this repository.

## 6. Model Evaluation

### 6.1 RQ1 — Effect of Training Dataset Size

Four independently trained models were evaluated using the same fixed test dataset containing 100 real and 100 fake images.

To evaluate the models, run:

```bash
python evaluate_models.py
```

The evaluation script loads the trained Keras models and predicts the classes of the test images.

Images are resized to the model input size of 224×224 pixels and normalized to the range from -1 to 1 before prediction.

Three evaluation metrics are reported:

- **Real Accuracy:** Percentage of real images correctly classified as real.
- **Fake Accuracy:** Percentage of fake images correctly classified as fake.
- **Overall Accuracy:** Percentage of all test images correctly classified.

Overall accuracy is calculated as:

**Overall Accuracy = (Correct Predictions / Total Test Images) × 100**

### 6.2 RQ2 — Effect of Image Quality

RQ2 investigates how reducing image resolution affects classification accuracy.

The trained **Model 500** was used for all image quality conditions. No additional model training was performed.

The same 200 test images were used in every condition.

Four image quality conditions were evaluated:

- Original image quality
- 128×128 resolution
- 64×64 resolution
- 32×32 resolution

To generate the reduced-resolution images, run:

```bash
python create_quality_dataset.py
```

The script downsamples the original test images to the specified resolutions and then resizes them back to their original dimensions using bicubic interpolation.

The processed images are saved in PNG format.

The following directories are created:

```text
test_128/
├── real/
└── fake/

test_64/
├── real/
└── fake/

test_32/
├── real/
└── fake/
```

Each directory contains 100 real and 100 fake images.

To evaluate the image quality conditions using Model 500, run:

```bash
python evaluate_quality.py
```

The script evaluates all four conditions and reports real-class, fake-class, and overall accuracy.

## 7. Experimental Results

### 7.1 RQ1 — Training Dataset Size

| Model | Real Accuracy | Fake Accuracy | Overall Accuracy |
|---|---:|---:|---:|
| Model 20 | 61.0% | 61.0% | 61.0% |
| Model 50 | 69.0% | 64.0% | 66.5% |
| Model 200 | 79.0% | 69.0% | 74.0% |
| Model 500 | 76.0% | 73.0% | 74.5% |

Overall accuracy increased from 61.0% with Model 20 to 74.5% with Model 500.

The improvement was relatively large when increasing the training dataset from 20 to 200 images per class.

However, increasing the dataset from 200 to 500 images per class improved overall accuracy by only 0.5 percentage points.

This suggests that increasing the training dataset size improved classification performance in this experiment, although the additional improvement became smaller at larger dataset sizes.

### 7.2 RQ2 — Image Quality

| Resolution | Real Accuracy | Fake Accuracy | Overall Accuracy |
|---|---:|---:|---:|
| Original | 76.0% | 73.0% | 74.5% |
| 128×128 | 73.0% | 68.0% | 70.5% |
| 64×64 | 87.0% | 45.0% | 66.0% |
| 32×32 | 99.0% | 0.0% | 49.5% |

Overall accuracy decreased from 74.5% for the original images to 49.5% for the 32×32 condition.

At 64×64 resolution, real-class accuracy increased to 87.0%, while fake-class accuracy decreased to 45.0%.

At 32×32 resolution, the model correctly classified 99% of real images but failed to correctly classify any fake images.

This indicates that the model became strongly biased toward predicting the real class when image resolution was severely reduced.

The results suggest that image quality affects not only overall classification accuracy but also the balance between real-class and fake-class predictions.

## 8. Project Structure

```text
PA04_Project/
├── README.md
├── environment.yml
├── .gitignore
├── prepare_dataset.py
├── create_quality_dataset.py
├── evaluate_models.py
├── evaluate_quality.py
│
├── trained_models/
│   ├── model_20/
│   │   ├── keras_model.h5
│   │   └── labels.txt
│   ├── model_50/
│   │   ├── keras_model.h5
│   │   └── labels.txt
│   ├── model_200/
│   │   ├── keras_model.h5
│   │   └── labels.txt
│   └── model_500/
│       ├── keras_model.h5
│       └── labels.txt
│
├── model_20/
├── model_50/
├── model_200/
├── model_500/
│
├── test_original/
├── test_128/
├── test_64/
├── test_32/
│
└── real_vs_fake/
    └── real-vs-fake/
```

The training, test, and original dataset directories are generated or downloaded locally and are excluded from Git tracking using `.gitignore`.

Only the Python scripts, environment configuration, documentation, and trained models are intended to be uploaded to GitHub.

## 9. Reproducibility and Limitations

### Reproducibility

A fixed random seed (`42`) was used for dataset selection.

The same test dataset was used for all four models in RQ1 and all four image quality conditions in RQ2.

The trained Keras models and evaluation scripts are provided to reproduce the evaluation procedure without retraining.

To reproduce the full experiment, users should download the original Kaggle dataset, generate the training subsets, train the four models in Teachable Machine, and run the evaluation scripts.

The dataset preparation script depends on the original dataset filenames and directory structure remaining unchanged.

### Limitations

The study used one training run for each dataset-size condition. Because model training involves randomness, repeating the training procedure may produce different results.

The experiment was conducted using one dataset containing FFHQ real faces and StyleGAN-generated faces. Therefore, the results cannot necessarily be generalized to other face datasets or AI image generation methods.

The image quality experiment only modified the test images. The model was not retrained using reduced-resolution training images.

The original image dataset is not redistributed in this repository. It is available through the Kaggle dataset link provided above.
