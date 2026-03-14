# 🐱🐶 CNN Image Classifier with Interpretability & Adversarial Robustness

A PyTorch CNN trained to classify cats vs. dogs, extended with **model interpretability** (Grad-CAM, Occlusion Sensitivity), **adversarial attack testing** (FGSM), and **vector similarity search** (Qdrant) — covering the full lifecycle of a production-minded deep learning project.

Built for NJIT CS370: Engineering AI Agents.

---

## What It Does

Four tasks in one notebook:

1. **CNN Classifier** — trains a custom convolutional network on 4,000 labeled images and evaluates with confusion matrix and precision-recall curve
2. **Model Interpretability** — uses Grad-CAM and Occlusion Sensitivity to visualize *what* the model focuses on when making predictions
3. **Adversarial Robustness** — applies Fast Gradient Sign Method (FGSM) attacks to measure how fragile the model is to input perturbations
4. **Vector Similarity Search** — extracts CNN embeddings and stores them in Qdrant to retrieve visually similar images at query time

---

## How It Works

### Task 1 — CNN Architecture

```
Input (3 × 64 × 64)
    ↓
Conv2d(3 → 16, 3×3) → ReLU → MaxPool(2×2)
    ↓
Conv2d(16 → 32, 3×3) → ReLU → MaxPool(2×2)
    ↓
Flatten → Linear(8192 → 128) → ReLU → Linear(128 → 1) → Sigmoid
```

- **Dataset**: `pantelism/cats-vs-dogs` (HuggingFace), 4,000 images, 80/20 train/val split
- **Loss**: Binary Cross Entropy (`BCELoss`)
- **Optimizer**: Adam (lr=0.001)
- **Epochs**: 10, batch size 32
- **Input size**: Images resized to 64×64, normalized to mean=0.5, std=0.5

### Task 2 — Interpretability

**Grad-CAM** (`captum.attr.LayerGradCam`):
- Computes gradients of the predicted class score with respect to the last conv layer (`conv_layers[3]`)
- Upsamples attribution map back to input resolution
- Overlays as a jet colormap heatmap on the original image — warm colors = high influence regions

**Occlusion Sensitivity** (`captum.attr.Occlusion`):
- Systematically slides a patch over the image, zeroing out each region
- Tests 3 window sizes: 10×10, 15×15, 20×20 (stride 8×8)
- Plots a hot colormap showing which regions most affect the prediction score when occluded

### Task 3 — FGSM Adversarial Attack

```python
perturbed_image = image + epsilon * sign(∇_x Loss)
```

- Computes gradient of BCE loss with respect to input pixels
- Adds a small perturbation in the direction that maximizes loss
- Tests epsilons: **0.01, 0.05, 0.1**
- Visualizes adversarial examples with true vs. predicted labels

### Task 4 — Qdrant Vector Search

- Extracts 128-dim embeddings from the CNN's penultimate layer (`fc_layers[1]`)
- Stores all 800 validation embeddings in an in-memory Qdrant collection (cosine distance)
- At query time: encodes a random image and retrieves top-5 most similar images by embedding similarity

---

## Tech Stack

| Component | Technology |
|---|---|
| **Deep Learning** | PyTorch |
| **Dataset** | HuggingFace Datasets (`pantelism/cats-vs-dogs`) |
| **Interpretability** | Captum (`LayerGradCam`, `Occlusion`) |
| **Adversarial Attacks** | FGSM (custom implementation) |
| **Vector Search** | Qdrant (in-memory) |
| **Evaluation** | scikit-learn (confusion matrix, classification report, precision-recall) |
| **Visualization** | Matplotlib, Seaborn |

---

## Results

### Classification (Task 1)
| Metric | Cat | Dog | Overall |
|---|---|---|---|
| Precision | 0.75 | 0.71 | — |
| Recall | 0.71 | 0.75 | — |
| F1-Score | 0.73 | 0.73 | — |
| **Accuracy** | — | — | **73%** |

Trained for 10 epochs on 3,200 images, validated on 800.

### Adversarial Robustness (Task 3)
| Epsilon | Test Accuracy |
|---|---|
| 0.01 | 49.25% |
| 0.05 | 49.25% |
| 0.1 | 49.25% |

Model accuracy drops to near-chance (49%) under even minimal FGSM perturbation, demonstrating vulnerability to adversarial examples.

### Vector Search (Task 4)
- 800 embeddings indexed in Qdrant
- Top-5 similar image retrieval using 128-dim cosine similarity
- Visually similar cats/dogs retrieved correctly across random query images

---

## Project Structure

```
cats-vs-dogs-cnn/
├── cnn_image_classifier.ipynb    # Full notebook: all 4 tasks with outputs
└── README.md
```

This project is self-contained in a single notebook. All code, outputs, and visualizations are inside `cnn_image_classifier.ipynb`.

---

## How to Run

### Prerequisites
- Python 3.11+
- GPU recommended but not required

### Setup
```bash
git clone https://github.com/AvanishRawat/cats-vs-dogs-cnn.git
cd cats-vs-dogs-cnn
pip install torch torchvision datasets captum qdrant-client scikit-learn matplotlib seaborn
```

### Run
Open the notebook in Jupyter or VS Code:
```bash
jupyter notebook cnn_image_classifier.ipynb
```

Then run all cells in order. The dataset downloads automatically from HuggingFace.

---

## License

MIT
