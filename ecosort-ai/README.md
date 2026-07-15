# ♻️ EcoSort AI — Intelligent Waste Classification System

A production-style deep learning project that classifies waste images into recyclable categories (**cardboard, glass, metal, paper, plastic, trash**) using transfer learning. Built to demonstrate an end-to-end ML workflow: data pipeline → training → evaluation → explainability → deployment.

---

## 🎯 Why this project (for recruiters/portfolio reviewers)

Most beginner portfolios show CIFAR-10/MNIST classifiers. This project instead demonstrates:

- **Real-world problem framing** — automated recycling sorting has genuine industry use (waste-management robotics, smart bins).
- **Transfer learning** with a pretrained CNN (ResNet18/EfficientNet-B0) instead of training from scratch.
- **Full MLOps-lite pipeline**: config-driven training, checkpointing, metrics logging, reproducibility.
- **Model explainability** via Grad-CAM (shows *why* the model made a decision — a big differentiator in interviews).
- **Deployment**: a working Streamlit web app for live inference, plus a CLI inference script.
- **Proper evaluation**: accuracy, precision/recall/F1 per class, confusion matrix — not just "accuracy: 95%".

---

## 🗂️ Project Structure

```
ecosort-ai/
├── README.md
├── requirements.txt
├── configs/
│   └── config.yaml            # all hyperparameters/paths in one place
├── src/
│   ├── dataset.py             # data loading, transforms, train/val/test split
│   ├── model.py                # model architecture (transfer learning setup)
│   ├── train.py                # training loop with checkpointing
│   ├── evaluate.py             # metrics, confusion matrix, classification report
│   ├── gradcam.py              # Grad-CAM explainability visualizations
│   └── predict.py              # single-image / batch inference CLI
├── app/
│   └── streamlit_app.py       # interactive web demo
├── models/                     # saved model checkpoints (.pth)
├── notebooks/
│   └── eda.ipynb              # exploratory data analysis (optional)
└── data/
    └── sample/                 # placeholder for dataset (see setup below)
```

---

## 📦 Dataset

This project is built around the **TrashNet / Garbage Classification** dataset (6 classes: cardboard, glass, metal, paper, plastic, trash), publicly available on Kaggle:

👉 https://www.kaggle.com/datasets/asdasdasasdas/garbage-classification

**Setup:**
1. Download and unzip the dataset.
2. Arrange it in `ImageFolder` format (one folder per class):
```
data/
├── cardboard/
├── glass/
├── metal/
├── paper/
├── plastic/
└── trash/
```
3. Update `data_dir` in `configs/config.yaml` to point to this folder.

> 💡 You can swap in **any** image classification dataset arranged this way — the code is dataset-agnostic. Great for reusing this template for a different domain (e.g., plant disease, food classification) in future portfolio pieces.

---

## ⚙️ Setup

```bash
git clone <your-repo-url>
cd ecosort-ai
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

---

## 🚀 Usage

### 1. Train the model
```bash
python src/train.py --config configs/config.yaml
```
Trains a ResNet18 (ImageNet-pretrained) with the last layers fine-tuned. Saves the best checkpoint to `models/best_model.pth` and logs metrics per epoch.

### 2. Evaluate the model
```bash
python src/evaluate.py --config configs/config.yaml --checkpoint models/best_model.pth
```
Outputs accuracy, per-class precision/recall/F1, and a confusion matrix plot (`models/confusion_matrix.png`).

### 3. Explain a prediction (Grad-CAM)
```bash
python src/gradcam.py --checkpoint models/best_model.pth --image path/to/image.jpg
```
Generates a heatmap overlay showing which pixels influenced the model's decision.

### 4. Run inference on a single image
```bash
python src/predict.py --checkpoint models/best_model.pth --image path/to/image.jpg
```

### 5. Launch the web demo
```bash
streamlit run app/streamlit_app.py
```
Upload an image in the browser and get instant predictions with confidence scores and a Grad-CAM overlay.

---

## 🧠 Model Architecture

- **Backbone:** ResNet18 pretrained on ImageNet (swap-in EfficientNet-B0 supported in `model.py`)
- **Head:** Global average pool → Dropout(0.3) → Linear(512, num_classes)
- **Training strategy:** Freeze backbone for first N epochs (warm-up), then unfreeze for full fine-tuning at a lower learning rate
- **Loss:** CrossEntropyLoss with label smoothing
- **Optimizer:** AdamW + cosine learning rate schedule
- **Augmentation:** random crop, horizontal flip, color jitter, rotation (via `torchvision.transforms`)

---

## 📊 Example Results (fill in with your run)

| Metric | Score |
|---|---|
| Test Accuracy | 92.4% |
| Macro F1 | 0.91 |
| Best Epoch | 18 |

*(Replace with your actual numbers after training — recruiters like seeing real, specific results, not placeholders.)*

---

## 🔭 Future Improvements

- Export to ONNX / TorchScript for faster inference
- Add a FastAPI backend for a production-style REST API
- Containerize with Docker for deployment
- Track experiments with Weights & Biases / MLflow
- Add active-learning loop for hard/misclassified examples

---

## 🛠️ Tech Stack

`Python` · `PyTorch` · `torchvision` · `scikit-learn` · `Streamlit` · `Grad-CAM` · `Matplotlib/Seaborn`

---

## 📄 License

MIT — free to use and adapt for your own portfolio.
