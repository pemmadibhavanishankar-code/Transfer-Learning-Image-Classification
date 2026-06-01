# Image Classification via Transfer Learning (ResNet18)

A PyTorch-based Deep Learning framework that leverages a pre-trained **ResNet18** backbone to classify a 3-class target subset (**Airplanes**, **Automobiles**, and **Birds**) from the CIFAR-10 dataset. 

---

## 📌 Project Overview

This project consists of an end-to-end computer vision pipeline split into two distinct operational steps:
1. **Transfer Learning Pipeline:** Freezing a pre-trained ResNet18 feature extractor, adapting its classification layer, training it on custom data, and exporting optimal state weights.
2. **Deterministic Inference Engine:** A standalone prediction script that reconstructs the network configuration, loads the trained weight vectors, and predicts the class of a local test image with explicit percentage confidence scores.

---

## 🛠️ Tech Stack & Requirements

The project utilizes the industry-standard **PyTorch** deep learning eco-system:
* **Deep Learning Engine:** `torch`, `torchvision`
* **Image Processing Engine:** `Pillow (PIL)`
* **Data Visualizations:** `matplotlib`, `numpy`

To set up your environment, run the following installation command:
```bash
pip install torch torchvision pillow matplotlib numpy
