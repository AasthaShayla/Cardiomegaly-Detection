# Cardiomegaly Detection - Chest X-ray Classification with Transfer Learning

A deep-learning image classifier that detects **cardiomegaly** (an abnormally
enlarged heart) from chest X-rays, using **transfer learning** on a pre-trained
**VGG16** convolutional network.

> Educational computer-vision project. Not a medical device - do not use for
> diagnosis.

## Problem

Cardiomegaly is often first spotted on a chest X-ray. This project frames it as a
binary image-classification task - **Normal** vs **Cardiomegaly** - to explore how
transfer learning performs on medical imaging with a modest dataset.

## Approach

- **Model:** VGG16 pre-trained on ImageNet as a feature extractor (top layers
  removed), with a `Flatten` + `Dense(softmax)` classification head added on top.
- **Input:** chest X-ray images resized to `128 × 128 × 3`, labels one-hot encoded.
- **Training:** categorical cross-entropy loss, Adam optimizer; data split into
  train / validation / test (≈ 4,136 / 492 / 516 images).
- **Inference:** a `predict()` helper loads a single X-ray and returns the
  predicted class with a confidence score.

## Results

| Split | Accuracy |
|-------|----------|
| Train | ~97% |
| Validation | ~79% |

The train/validation gap reflects the small validation set and the usual
overfitting risk of transfer learning on limited medical data - a good starting
point for regularization, augmentation, and fine-tuning experiments.

## Demo

A short screen recording of the model classifying X-rays is included:
[`cardiomegaly_detection.mp4`](./cardiomegaly_detection.mp4).

## Tech stack

Python · TensorFlow / Keras · VGG16 · NumPy · pandas · Matplotlib · seaborn ·
Jupyter Notebook.

## Run it

1. Get a chest X-ray dataset organized as `train/ val/ test/` with `NORMAL/` and
   `CARDIOMEGALY/` subfolders (e.g. from Kaggle) and update the `path` variable in
   the notebook.
2. Install dependencies:
   ```bash
   pip install tensorflow keras numpy pandas matplotlib seaborn
   ```
3. Open `Cardiomegaly_detection.ipynb` in Jupyter and run the cells top to bottom.

## Possible improvements

- Data augmentation + class balancing
- Fine-tuning the top VGG16 blocks
- Grad-CAM heatmaps to visualize what the model looks at
- Reporting precision/recall/AUC, not just accuracy
