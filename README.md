# 🧠 ML Basics Showcase: Logistic Regression & MLP Classifiers

This project is to demonstrate foundational classification workflows using Logistic Regression and Multi-Layer Perceptron (MLP) on benchmark datasets like Fashion-MNIST and CIFAR-10. It is designed as an introduction to deep learning curriculum to highlight knowledge of key machine learning concepts through hands-on training, evaluation, and visualization.

---

## 📁 Project Structure

```bash
intro-to-ml-basics/
├── scripts/              # Reusable Python modules
│   ├── dataset_loader.py
│   ├── dying_neuron_test.py
│   ├── loss_surface_plotting.py
│   └── model_loader.py
├── notebooks/            # Experiments and visualizations
│   ├── logistic_regression_performance.ipynb
│   └── classification_with_MLP.ipynb
├── requirements.txt
├── config.yaml
└── README.md
```

---

## 📌 Objectives

* Implement logistic regression and MLP from scratch using Tensorflow
* Train on Fashion-MNIST/CIFAR datasets
* Visualize training and validation performance
* Understand underfitting vs overfitting
* Apply techniques like initialization, regularization and dropout

---

## 🛠️ Models Implemented

| Model               | Dataset       | Key Features                              |
| ------------------- | ------------- | ----------------------------------------- |
| Logistic Regression |  FMNIST       | Linear classifier with Softmax activation |
| MLP (single layer)  | FMNIST, CIFAR | ReLU, Softmax, Early Stopping             |
| MLP (3 layers)      | FMNIST, CIFAR | He Normal initialization, Leaky ReLU, 
                                        Dropout, Softmax, L2 regularization       |

---

## 📊 Results

Here are the Performance metrics and training/validation curves:

| Model               | Dataset | Accuracy | Notes               |
| ------------------- | ------- | -------- | ------------------- |
| Logistic Regression | FMNIST | \~84.59% | Simple linear model |
 (w\o Regularization) |
| ------------------- | ------- | -------- | ------------------- |
| Logistic Regression | 
 (l2 Regularization & | FMNIST  | \~81.92% | Impose Prior Bias & |
  Early callback)     |                      Stop Overfitting
| ------------------- | ------- | -------- | ------------------- |
| MLP (1-layer)       | FMNIST  | \~88.26% | Better nonlinearity |
                      | CIFAR   | \~46.95% | Underfit model      |
| ------------------- | ------- | -------- | ------------------- |
| MLP (3-layers,      | FMNIST  | \~88.46% | Similar Performance |
       & Dropout)     | CIFAR   | \~57.34% | Struggle to fit data|

> 📈 Training & validation accuracy, most relevant pixels, and loss curves are visualized in the notebooks under `/notebooks`.

---

## 🔍 Key Takeaways

* Logistic regression can perform relatively well on simpler datasets like FMNIST.
* MLPs can model more complex decision boundaries with hidden layers but still struggles with more complex data like RGB images in CIFAR-10
* Dropout and L2 regularization are helpful in reducing overfitting, especially for deeper models.
* Proper parameter initialization helps avoid vanishing and exploding gradients.

---

## 🚀 Setup Instructions

### 1. Clone the repository

```bash
git https://github.com/Salekin-13/intro-to-ml-basics.git
cd intro-to-ml-basics
```

### 2. Set up environment

```bash
pip install -r requirements.txt
```

---

## 📎 Notes

* Built with educational clarity in mind — every function and model is well-commented and structured
* Easy to extend with CNN or other datasets
* No external training libraries used — just Tensorflow and NumPy

---

## 💡 Future Work

* Add CNN-based classifier
* Add noise to input images and measure the change in accuracy

---

## 🙋‍♀️ Author

Made by \[Sumaiya Salekin]
Inspired by classic ML fundamentals 🌱
