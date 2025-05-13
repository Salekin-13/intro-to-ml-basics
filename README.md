# 🧠 ML Basics Showcase: Logistic Regression & MLP Classifiers

This project demonstrates foundational classification workflows using Logistic Regression and Multi-Layer Perceptron (MLP) on standard image datasets like MNIST and Fashion-MNIST. It is designed as part of a 2-week introductory deep learning curriculum to highlight key machine learning concepts through hands-on training, evaluation, and visualization.

---

## 📁 Project Structure

```bash
intro-to-ml-basics/
├── scripts/              # All reusable Python modules
│   ├── data_utils.py
│   ├── models.py
│   └── train.py
├── notebooks/            # Interactive experiments and visualizations
│   ├── 01_logistic_regression.ipynb
│   └── 02_mlp_classifier.ipynb
├── results/              # Saved plots and trained model files
├── requirements.txt
├── README.md
└── setup.py
```

---

## 📌 Objectives

* Implement logistic regression and MLP from scratch using PyTorch
* Train on MNIST/Fashion-MNIST datasets
* Visualize training and validation performance
* Understand underfitting vs overfitting
* Apply techniques like regularization and dropout

---

## 🛠️ Models Implemented

| Model               | Dataset       | Key Features                              |
| ------------------- | ------------- | ----------------------------------------- |
| Logistic Regression | MNIST, FMNIST | Linear classifier                         |
| MLP (2–3 layers)    | MNIST, FMNIST | ReLU, Dropout, Softmax, L2 regularization |

---

## 📊 Results

Here are sample training/validation curves and performance metrics:

| Model               | Dataset | Accuracy | Notes               |
| ------------------- | ------- | -------- | ------------------- |
| Logistic Regression | MNIST   | \~92%    | Simple linear model |
| MLP (2-layer)       | MNIST   | \~97%    | Better nonlinearity |
| MLP (with dropout)  | FMNIST  | \~89%    | Reduced overfitting |

> 📈 Training & validation accuracy/loss curves are visualized in the notebooks under `/notebooks`.

---

## 🔍 Key Takeaways

* Logistic regression can perform well on simpler datasets like MNIST but struggles with more complex patterns.
* MLPs can model more complex decision boundaries with hidden layers.
* Dropout and L2 regularization are helpful in reducing overfitting, especially for deeper models.

---

## 🚀 Setup Instructions

### 1. Clone the repository

```bash
git clone https://github.com/yourusername/intro-to-ml-basics.git
cd intro-to-ml-basics
```

### 2. Set up environment

```bash
pip install -e .
pip install -r requirements.txt
```

---

## ✅ Requirements

Python 3.8+
Recommended packages:

```
torch
torchvision
numpy
matplotlib
scikit-learn
tqdm
```

(Already included in `requirements.txt`)

---

## 📎 Notes

* Built with educational clarity in mind — every function and model is well-commented and structured
* Easy to extend with CNN or other datasets
* No external training libraries used — just PyTorch and NumPy

---

## 💡 Future Work

* Add CNN-based classifier
* Train on CIFAR-10
* Add TensorBoard integration
* Visualize learned weights

---

## 🙋‍♀️ Author

Made with care by \[Your Name]
Inspired by classic ML fundamentals + curiosity 🌱
