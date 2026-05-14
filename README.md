<div align="center">

# 🛡️ Intelligent Banking Fraud Detection System

### Using Data Science & Deep Learning

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white)](https://tensorflow.org)
[![Keras](https://img.shields.io/badge/Keras-Deep%20Learning-D00000?style=for-the-badge&logo=keras&logoColor=white)](https://keras.io)
[![Colab](https://img.shields.io/badge/Google%20Colab-Notebook-F9AB00?style=for-the-badge&logo=google-colab&logoColor=white)](https://colab.research.google.com)
[![Kaggle Dataset](https://img.shields.io/badge/Kaggle-Dataset-20BEFF?style=for-the-badge&logo=kaggle&logoColor=white)](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
[![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)](LICENSE)

<br>

> **An end-to-end machine learning pipeline for real-time detection of fraudulent credit card transactions, built as an academic project combining Data Science and Advanced AI methodologies.**

<br>

<img src="https://img.shields.io/badge/Accuracy-High%20Performance-brightgreen?style=flat-square" alt="accuracy"/>
<img src="https://img.shields.io/badge/Transactions-284%2C807-blue?style=flat-square" alt="transactions"/>
<img src="https://img.shields.io/badge/Fraud%20Cases-492-red?style=flat-square" alt="fraud cases"/>
<img src="https://img.shields.io/badge/Features-30-purple?style=flat-square" alt="features"/>

</div>

---

## 📑 Table of Contents

- [Overview](#-overview)
- [Key Features](#-key-features)
- [Tech Stack](#-tech-stack)
- [Dataset](#-dataset)
- [Project Architecture](#-project-architecture)
- [Pipeline Workflow](#-pipeline-workflow)
  - [1. Data Collection](#1--data-collection)
  - [2. Data Cleaning](#2--data-cleaning)
  - [3. Exploratory Data Analysis](#3--exploratory-data-analysis-eda)
  - [4. Data Preparation](#4--data-preparation)
  - [5. Deep Learning Model](#5--deep-learning-model-ann)
  - [6. Prediction](#6--prediction)
  - [7. Evaluation](#7--evaluation)
  - [8. Decision Layer](#8--decision-layer)
- [Model Architecture](#-model-architecture)
- [Results](#-results)
- [Business Value](#-business-value)
- [Getting Started](#-getting-started)
- [Project Structure](#-project-structure)
- [Academic Context](#-academic-context)
- [Author](#-author)

---

## 🎯 Overview

Fraudulent transactions cost the global banking industry **billions of dollars annually**. This project tackles that challenge by building an **Intelligent Fraud Detection System** using a complete Data Science pipeline combined with Deep Learning.

The system processes **284,807 real-world credit card transactions**, identifies patterns in the data, and uses an **Artificial Neural Network (ANN)** to classify transactions as **legitimate or fraudulent** — with high accuracy.

```
📊 Data  →  🧹 Clean  →  🔍 Understand  →  🧠 Model  →  📈 Predict  →  ✅ Decide
```

This project spans two academic modules:

| Module | Focus |
|--------|-------|
| **Science de Données** | Data pipeline, EDA, preprocessing, evaluation |
| **Intelligence Artificielle Avancée** | ANN architecture, Deep Learning, optimization |

---

## ✨ Key Features

| Feature | Description |
|---------|-------------|
| 🔄 **End-to-End Pipeline** | Complete workflow from raw data to business decisions |
| 🧹 **Robust Data Cleaning** | Duplicate removal, missing value handling, quality validation |
| 📊 **Rich Visualizations** | Distribution plots, correlation heatmaps, fraud analysis charts |
| 🧠 **Deep Learning Model** | Custom ANN with optimized architecture for binary classification |
| 📈 **Comprehensive Evaluation** | Confusion matrix, classification report, accuracy/loss curves |
| 🏦 **Business Decision Layer** | Transforms predictions into actionable banking decisions |

---

## 🛠️ Tech Stack

<div align="center">

| Category | Technologies |
|----------|-------------|
| **Language** | ![Python](https://img.shields.io/badge/Python-3776AB?style=flat-square&logo=python&logoColor=white) |
| **Environment** | ![Google Colab](https://img.shields.io/badge/Google%20Colab-F9AB00?style=flat-square&logo=google-colab&logoColor=white) |
| **Data Processing** | ![Pandas](https://img.shields.io/badge/Pandas-150458?style=flat-square&logo=pandas&logoColor=white) ![NumPy](https://img.shields.io/badge/NumPy-013243?style=flat-square&logo=numpy&logoColor=white) |
| **Visualization** | ![Matplotlib](https://img.shields.io/badge/Matplotlib-11557C?style=flat-square&logo=matplotlib&logoColor=white) ![Seaborn](https://img.shields.io/badge/Seaborn-3776AB?style=flat-square&logo=python&logoColor=white) |
| **Machine Learning** | ![Scikit-learn](https://img.shields.io/badge/Scikit--learn-F7931E?style=flat-square&logo=scikit-learn&logoColor=white) |
| **Deep Learning** | ![TensorFlow](https://img.shields.io/badge/TensorFlow-FF6F00?style=flat-square&logo=tensorflow&logoColor=white) ![Keras](https://img.shields.io/badge/Keras-D00000?style=flat-square&logo=keras&logoColor=white) |

</div>

---

## 📦 Dataset

| Property | Value |
|----------|-------|
| **Source** | [Credit Card Fraud Detection — Kaggle](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) |
| **Total Transactions** | 284,807 |
| **Fraudulent Cases** | 492 (0.17%) |
| **Features** | 30 columns (V1–V28 via PCA + `Time` + `Amount`) |
| **Target Variable** | `Class` (0 = Legitimate, 1 = Fraud) |
| **Challenge** | Highly imbalanced dataset |

> ⚠️ **Note:** The dataset is **extremely imbalanced** — only **0.17%** of transactions are fraudulent, making this a challenging classification problem that requires careful handling.

---

## 🏗️ Project Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    FRAUD DETECTION SYSTEM                       │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌──────────┐   ┌──────────┐   ┌──────────┐   ┌──────────┐    │
│  │   DATA   │──▸│  CLEAN   │──▸│   EDA    │──▸│  PREP    │    │
│  │COLLECTION│   │& VALIDATE│   │& VISUALIZE│  │& SCALE   │    │
│  └──────────┘   └──────────┘   └──────────┘   └──────────┘    │
│                                                     │          │
│                                                     ▼          │
│  ┌──────────┐   ┌──────────┐   ┌──────────┐   ┌──────────┐    │
│  │ DECISION │◂──│ EVALUATE │◂──│ PREDICT  │◂──│  TRAIN   │    │
│  │  LAYER   │   │& METRICS │   │  (ANN)   │   │  MODEL   │    │
│  └──────────┘   └──────────┘   └──────────┘   └──────────┘    │
│       │                                                        │
│       ▼                                                        │
│  ┌──────────────────────────────────────────┐                  │
│  │  ✅ Validate  │  ⚠️ Verify  │  🚫 Block  │                  │
│  └──────────────────────────────────────────┘                  │
└─────────────────────────────────────────────────────────────────┘
```

---

## 🔄 Pipeline Workflow

### 1. 📥 Data Collection

Loading the Kaggle credit card dataset containing **284,807 real-world transactions** with 30 engineered features (PCA-transformed for anonymization).

```python
import pandas as pd
df = pd.read_csv('creditcard.csv')
print(f"Shape: {df.shape}")  # (284807, 31)
```

---

### 2. 🧹 Data Cleaning

Ensuring data quality through systematic validation:

- ✅ Detection & removal of duplicate records (`drop_duplicates()`)
- ✅ Missing value analysis (`isnull().sum()`)
- ✅ Data type validation & consistency checks
- ✅ Quality assurance before model training

> 💡 **Principle:** *Garbage In → Garbage Out* — Clean data is the foundation of reliable predictions.

---

### 3. 🔍 Exploratory Data Analysis (EDA)

Understanding data distribution and patterns through visualization:

| Visualization | Purpose |
|--------------|---------|
| **Class Distribution** (`countplot`) | Reveals severe class imbalance |
| **Amount Histogram** | Shows transaction amount distribution |
| **Correlation Heatmap** | Identifies feature relationships |
| **Fraud Analysis** | Characterizes fraudulent transaction patterns |

**Key Findings:**
- 🔴 Extreme class imbalance (99.83% vs 0.17%)
- 🔴 Fraudulent transactions tend to have lower amounts
- 🔴 Strong correlations exist between certain PCA features

---

### 4. ⚙️ Data Preparation

Preparing the data for optimal model performance:

```python
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Feature / Target separation
X = df.drop('Class', axis=1)
y = df['Class']

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Feature Scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
```

---

### 5. 🧠 Deep Learning Model (ANN)

Building an **Artificial Neural Network** for binary classification:

```python
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

model = Sequential([
    Dense(32, activation='relu', input_shape=(X_train.shape[1],)),
    Dense(16, activation='relu'),
    Dense(1, activation='sigmoid')
])

model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

history = model.fit(X_train, y_train, epochs=10, batch_size=32, validation_split=0.2)
```

<details>
<summary><b>📐 Model Summary</b></summary>

| Layer | Neurons | Activation | Purpose |
|-------|---------|------------|---------|
| Input | 30 | — | Receives scaled features |
| Hidden 1 | 32 | ReLU | Non-linear feature extraction |
| Hidden 2 | 16 | ReLU | Higher-level pattern learning |
| Output | 1 | Sigmoid | Fraud probability (0–1) |

**Hyperparameters:**

| Parameter | Value |
|-----------|-------|
| Optimizer | Adam |
| Loss Function | Binary Crossentropy |
| Epochs | 10 |
| Batch Size | 32 |
| Validation Split | 20% |

</details>

---

### 6. 🎯 Prediction

The trained model outputs a **fraud probability score** for each transaction:

```
Probability ≥ 0.5  →  🚨 Fraudulent
Probability < 0.5  →  ✅ Legitimate
```

---

### 7. 📊 Evaluation

Comprehensive model assessment using multiple metrics:

| Metric | Description |
|--------|-------------|
| **Classification Report** | Precision, Recall, F1-Score per class |
| **Confusion Matrix** | True/False Positives & Negatives |
| **Accuracy Score** | Overall prediction accuracy |
| **Accuracy/Loss Curves** | Training convergence visualization |

---

### 8. 🏦 Decision Layer

Transforming raw predictions into **actionable business decisions**:

```
┌─────────────────────────────────────────────────────────┐
│              DECISION ENGINE                             │
├──────────────┬──────────────────────┬───────────────────┤
│  Score < 0.3 │  0.3 ≤ Score < 0.7  │   Score ≥ 0.7     │
├──────────────┼──────────────────────┼───────────────────┤
│ ✅ APPROVED   │ ⚠️ VERIFY CLIENT     │ 🚫 BLOCKED        │
│ Transaction  │ Additional           │ Transaction       │
│ validated    │ authentication       │ suspended         │
└──────────────┴──────────────────────┴───────────────────┘
```

---

## 🧬 Model Architecture

```
          ┌─────────────────────────┐
          │     INPUT LAYER         │
          │   (30 features)         │
          └───────────┬─────────────┘
                      │
                      ▼
          ┌─────────────────────────┐
          │   HIDDEN LAYER 1        │
          │   32 neurons • ReLU     │
          └───────────┬─────────────┘
                      │
                      ▼
          ┌─────────────────────────┐
          │   HIDDEN LAYER 2        │
          │   16 neurons • ReLU     │
          └───────────┬─────────────┘
                      │
                      ▼
          ┌─────────────────────────┐
          │    OUTPUT LAYER         │
          │   1 neuron • Sigmoid    │
          │   P(Fraud) ∈ [0, 1]    │
          └─────────────────────────┘
```

---

## 📈 Results

The model demonstrates **strong performance** on the credit card fraud detection task:

| Aspect | Observation |
|--------|-------------|
| ✅ **Accuracy** | High overall accuracy on test set |
| ✅ **Fraud Detection** | Effective identification of fraudulent transactions |
| ✅ **Generalization** | Low overfitting, stable validation metrics |
| ✅ **Convergence** | Smooth accuracy/loss curves during training |

---

## 💼 Business Value

<div align="center">

| Benefit | Impact |
|---------|--------|
| 💰 **Reduce Financial Losses** | Prevent millions in fraud-related damages |
| 🔒 **Enhance Transaction Security** | Real-time protection for customers |
| ⚡ **Automate Fraud Detection** | Replace manual review processes |
| 🧠 **Intelligent Decision Making** | Data-driven risk assessment |
| 📈 **Scalable Architecture** | Handle growing transaction volumes |

</div>

**Applicable to:** Banks • Payment Platforms • Fintech Companies • Anti-Fraud Systems

---

## 🚀 Getting Started

### Prerequisites

```bash
Python >= 3.10
pip (Python package manager)
```

### Installation

```bash
# 1. Clone the repository
git clone https://github.com/your-username/banking-fraud-detection.git
cd banking-fraud-detection

# 2. Install dependencies
pip install pandas numpy matplotlib seaborn scikit-learn tensorflow

# 3. Download the dataset
# Visit: https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud
# Place creditcard.csv in the project root

# 4. Run the notebook
# Open in Google Colab or Jupyter Notebook
```

### Quick Start with Google Colab

1. Upload the notebook to [Google Colab](https://colab.research.google.com)
2. Upload `creditcard.csv` or mount Google Drive
3. Run all cells sequentially
4. View results and visualizations

---

## 📁 Project Structure

```
📦 banking-fraud-detection/
├── 📂 cours/
│   ├── 📂 AI avancee/          # Advanced AI course materials
│   │   ├── Chapitre 1.pdf
│   │   ├── Chapitre 2.pdf
│   │   ├── Chapitre 2-continuation.pdf
│   │   └── Chapitre 4.pdf
│   └── 📂 data/                # Data Science course materials
│       ├── Chapter4.pdf
│       ├── Chapter5.pdf
│       ├── Chapter6.pdf
│       ├── Chapter8.pdf
│       └── Chapter10.pdf
├── 📂 creditcard.csv/
│   └── creditcard.csv          # Main dataset (150MB)
├── 📄 creditcard.csv.zip       # Compressed dataset
├── 📄 guide_rapport_ia_avancee.md
├── 📄 guide_rapport_science_de_donnees.md
└── 📄 README.md                # ← You are here
```

---

## 🎓 Academic Context

| Detail | Information |
|--------|-------------|
| **Project** | Intelligent Banking Fraud Detection System |
| **Modules** | Science de Données • Intelligence Artificielle Avancée |
| **Institution** | École Polytechnique d'Agadir |
| **Academic Year** | 2025–2026 |
| **Program** | Engineering — Big Data & AI |

---

## 👤 Author

<div align="center">

### **Yassine Rachid**

🎓 Engineering Student — Big Data & Artificial Intelligence

🏫 École Polytechnique d'Agadir

[![GitHub](https://img.shields.io/badge/GitHub-181717?style=for-the-badge&logo=github&logoColor=white)](https://github.com/Hardrach)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-0A66C2?style=for-the-badge&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/yassine-rachid-b27aa6225/)

</div>

---

<div align="center">

**⭐ If you found this project useful, please consider giving it a star!**

<br>

Made with ❤️ and 🧠 by Yassine Rachid

<br>

![Visitors](https://api.visitorbadge.io/api/visitors?path=your-username%2Fbanking-fraud-detection&label=Visitors&countColor=%23263759)

</div>
