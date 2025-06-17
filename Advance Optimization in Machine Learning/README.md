# Advanced Optimization in Deep Learning

This repository provides structured materials for a hands-on tutorial series on advanced optimization techniques in deep learning. It balances theoretical foundations (~30 minutes) with hands-on practical implementations. It includes theoretical explanations, algorithm implementations, comparative studies, and deployment-aware strategies.

> ⚙️ *The tutorial is divided into four parts: foundational theory, core method implementation, comparative experiments, and deployment-oriented optimization.*

---

## 📁 Contents

### 1. [Part 1 - Optimization Techniques in Machine Learning](Part%201%20-%20Optimization%20Techniques%20in%20Machine%20Learning)

**Focus**: *Theoretical overview of classical and modern optimization algorithms in machine learning.*

Topics:
- Why Optimization Matters in ML
- Gradient Based Optimization
- Second-Order & Advanced Methods
- Hyperparameter Optimization
---

### 2. [Part 2 - Practical Implementation of Optimization Methods.ipynb](Part%202%20-%20Practical%20Implementation%20of%20Optimization%20Methods.ipynb)

**Focus**: *From-scratch implementation of core methods using NumPy.*

#### 🧩 Table of Contents:
1. [Simple NumPy Neural Network](#NumpyNeuralNetwork)  
2. [Activation Functions: ReLU, Tanh](#ActivationFunctions)  
3. [Gradient Descent Variants](#GradientDescent)  
4. [Linear Quantization](#Quantization)  
5. [Framework Demo: Single-line TensorFlow Comparison](#WithTensorflow)

<!-- Colab Badge -->
<a target="_blank" href="https://colab.research.google.com/drive/1pgsMo4c0Iax79YITcj1xiocbarkBhRLu?usp=sharing">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
</a>

---

### 3. [Part 3 - Advanced Optimization ML.ipynb](Part%203%20-%20Advanced%20Optimization%20ML.ipynb)

**Focus**: *Comparative optimizer performance, scheduling strategies, and regularization.*

#### 🎯 Learning Objectives:
- Understand optimizer dynamics and trade-offs (SGD, Adam, RMSProp).
- Implement & compare optimizers using Keras.
- Apply learning rate schedules: `ExponentialDecay`, `CosineDecay`, `OneCycleLR`.
- Use dropout, L2 regularization, and visualize their effects.
- Tune hyperparameters using Optuna.
- Engage with guided optimization challenges.

<!-- Colab Badge -->
<a target="_blank" href="https://colab.research.google.com/drive/1Hq3vD2aYJdUkIYZsUGxagkVT2EqXBY95?usp=sharing">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
</a>

---

### 4. [Part 4 - Neural Architecture Search & Deployment Optimization.ipynb](Part%204%20-%20Neural%20Architecture%20Search%20%26%20Deployment%20Optimization.ipynb)

**Focus**: *Optimization beyond training—design, compression, and deployment strategies.*

#### 🧠 Table of Contents:
1. [Introduction](#Introduction)  
2. [Imports & Dataset Overview](#Imports)  
3. [Neural Architecture Search (NAS)](#NAS)  
4. [Post-Training Quantization](#Code)  
5. [Quantization-Aware Training (QAT)](#QAT)  
6. [Conclusion](#Conclusion)  
7. [References](#References)

<!-- Colab Badge -->
<a target="_blank" href="https://colab.research.google.com/drive/10VieqNeG6X8nnb-s2BjWEgYC9A4CMkHO?usp=sharing">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
</a>

---

### 5. [requirements.txt](requirements.txt)

A list of Python package dependencies required to run the notebooks smoothly.

```bash
pip install -r requirements.txt
