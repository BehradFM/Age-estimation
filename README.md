# 🎯 Age Estimation using PyTorch

## 📌 Overview

This repository presents a deep learning-based **Age Estimation system** implemented with **PyTorch**. The goal of this project is to accurately predict human age from facial images using convolutional neural networks (CNNs) Model **(Resnet50)**.

Age estimation is a challenging computer vision task due to variations in lighting, pose, ethnicity, and facial attributes ([GitHub][1]). This project addresses these challenges by leveraging multiple large-scale datasets and modern deep learning techniques.

---

## 🚀 Features

* 🧠 Deep learning-based age prediction using PyTorch
* 📊 Support for multiple benchmark datasets
* 🔄 End-to-end pipeline: preprocessing, training, evaluation
* 🧩 Modular and extensible code structure
* ⚡ Easy experimentation with different models and parameters

---

## 🗂️ Datasets

This project utilizes several widely-used datasets for age estimation:

* **IMDB Dataset** – Large-scale dataset of celebrity faces
* **WIKI Dataset** – Companion dataset with labeled age data
* **UTKFace Dataset** – Diverse dataset with age, gender, and ethnicity labels
* **Morph II Dataset** – High-quality longitudinal face dataset

📥 **Note:**
Datasets are not included in this repository due to size limitations. Please download them manually and organize them according to the expected directory structure.

---

## 🏗️ Project Structure

```
.
├── image/        
├── models/          
├── age_estimation.ipynb
├── requirements.txt # Dependencies
└── README.md
```

---

## Pipeline Overview

### 1. Face Detection & Cropping

- A YOLOv8 face detection model (`yolov8n-face.pt`) is used to detect faces.
- For each image, the largest detected face is selected and expanded by a margin (default 20%).
- Cropped faces are saved to `cropped_faces/` with a unique filename based on the original path.

### 2. Data Cleaning & EDA

- Images with dimensions smaller than 100×100 pixels are filtered out.
- Age values outside the range [1, 80] are removed.
- For UTKFace, gender and ethnicity are also extracted.
- Exploratory visualizations (histograms, violin plots, random sample grids) are generated.

### 3. Model Training

- **Base model:** ResNet50 pre-trained on ImageNet, with the final fully connected layer replaced by a single neuron for age regression.
- **Loss function:** Mean Squared Error (MSE)
- **Optimizer:** Adamw
- **Evaluation metric:** Mean Absolute Error (MAE)
- The training loop uses `AverageMeter` to track loss and metric over epochs.

### 4. Inference

- The trained model can be used to predict age on new images.
- Optionally, face detection is applied before feeding the cropped face to the model.
- A helper function `inference()` returns the predicted age and the cropped PIL image.

## ⚙️ Installation

Clone the repository:

```bash
git clone https://github.com/BehradFM/Age-estimation.git
cd Age-estimation
```

required packages:

```bash
pip install -r requirements.txt
```

Datasets:

[IMDB and Wiki Dataset](https://data.vision.ee.ethz.ch/cvl/rrothe/imdb-wiki/)

[UTKFace](https://susanqq.github.io/UTKFace/)

[Morph 2](https://www.kaggle.com/datasets/chiragsaipanuganti/morph)

---

## 🧠 Model Architecture

The project is based on **Convolutional Neural Networks (CNNs)**, which are widely used for image-based tasks such as age estimation. These models learn hierarchical facial features to estimate age effectively.

---
## ## Training Strategy

The age estimation model is trained using a **staged transfer learning approach**. We start from a ResNet50 model pre‑trained on ImageNet and progressively fine‑tune it on increasingly larger and more diverse face datasets. This strategy allows the model to first learn general facial features and then adapt to the specific task of age regression.

### Stage 1: Pre‑training on IMDB‑Wiki (Coarse Tuning)

The IMDB‑Wiki dataset is large but contains noisy labels (ages estimated from birth and photo dates). We first train the model on **IMDB** for 15 epochs with a relatively high learning rate (`1e-3`) to quickly adapt the pre‑trained features to facial age patterns. Then, we continue training on **Wiki** for 30 epochs with a much lower learning rate (`1e-5`). The lower LR on Wiki helps refine the model without overfitting to the smaller, cleaner dataset.

### Stage 2: Fine‑tuning on UTKFace

UTKFace provides high‑quality age labels (directly annotated) and a balanced age distribution. We load the model trained on Wiki and fine‑tune it on UTKFace for 50 epochs with an even smaller learning rate (`1e-5`). This allows the model to specialize on precise age regression while preserving previously learned representations.

### Stage 3: Final Fine‑tuning (Combined Data)

Finally, we combine all training data (IMDB, Wiki, UTKFace) and perform a final fine‑tuning for 30 epochs with a tiny learning rate (`5e-6`). This stage harmonises the model across all datasets, reducing domain‑specific biases and improving generalisation.

Across all stages, we use:
- **Loss function:** Mean Absolute Error (MAE) .
- **Optimizer:** Adamw with weight decay (`1e-4`) to prevent overfitting.

This staged curriculum mimics a coarse‑to‑fine learning schedule and yields a robust age estimator.

### Hyperparameter Summary

| Dataset     | Learning Rate (LR) | Weight Decay (WD) | Epochs |
|-------------|--------------------|-------------------|--------|
| IMDB        | 1e-3               | 1e-4              | 15     |
| Wiki        | 1e-5               | 1e-4              | 30     |
| UTKFace     | 1e-5               | 1e-4              | 50     |
| Final (All) | 5e-6               | 1e-4              | 30     |

> All trainings use the same Adam optimizer and MSE loss. The final model is the one fine‑tuned on the combined dataset.
---

## 📊 Evaluation Metrics

The performance of the model is evaluated using:

* **Mean Absolute Error (MAE)** – measures prediction accuracy
* **Accuracy** (optional) – for age group classification

---

## 🔧 Configuration

You can configure key hyperparameters via command-line arguments:

* Learning rate
* Batch size
* Number of epochs
* Dataset selection

---

## 🤝 Contributing

Contributions are welcome!

1. Fork the repository
2. Create a new branch
3. Commit your changes
4. Open a Pull Request

---

## 📜 License

This project is released under the MIT License.

---

## 📬 Contact

If you have any questions, suggestions, or issues, feel free to open an issue in the repository.

---

## ⭐ Acknowledgements

* IMDB-WIKI dataset creators
* UTKFace dataset
* Morph II dataset
* Open-source deep learning community

---

💡 If you find this project useful, consider giving it a ⭐ on GitHub!
