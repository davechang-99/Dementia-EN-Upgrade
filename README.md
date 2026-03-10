
# 🧠 Speech-Based Dementia Detection using Multimodal AI
## Acoustic, Linguistic, and Transformer-Based Modeling for Early Cognitive Screening

[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)]
[![PyTorch](https://img.shields.io/badge/PyTorch-DeepLearning-red)]
[![License](https://img.shields.io/badge/License-MIT-green)]
[![Research](https://img.shields.io/badge/Research-AI%20Healthcare-purple)]
[![Status](https://img.shields.io/badge/Status-Research%20Prototype-orange)]

---

# 📄 Paper-style Overview

This repository presents a **speech-based dementia detection framework** designed for research comparable to projects released with **top AI conferences such as NeurIPS, ICML, and ACL**.

The system investigates whether **spontaneous speech signals contain reliable digital biomarkers of cognitive decline**.

The proposed pipeline integrates:

- **Acoustic signal processing**
- **Linguistic analysis**
- **Temporal speech biomarkers**
- **Classical machine learning**
- **Transformer-based deep learning**

Experiments are conducted using the **ADReSS 2020 dementia detection benchmark dataset**.

Model performance is evaluated with **5‑fold cross-validation** and **statistical significance testing using the Friedman test**.

The ultimate goal is to explore **speech as a scalable, non‑invasive biomarker for early dementia screening**.

---

# 🎯 Research Contributions

This project provides the following contributions:

### 1️⃣ Multimodal Speech Feature Modeling
Combines:

- acoustic biomarkers
- linguistic complexity metrics
- temporal speech behavior

### 2️⃣ Classical vs Deep Learning Comparison

Benchmarking:

- Logistic Regression
- Support Vector Machine
- Random Forest
- Vision Transformer

### 3️⃣ Transformer-based Speech Representation

Mel‑spectrogram images are processed using a **Vision Transformer architecture**.

### 4️⃣ Statistical Model Comparison

Uses **Friedman non‑parametric testing** for robust comparison across cross‑validation folds.

### 5️⃣ Cognitive Score Prediction

Explores predicting **MMSE scores using regression models**.

---

# 🧩 System Architecture

```
Speech Recording
      │
      ▼
Audio Processing
(librosa)
      │
      ▼
Feature Extraction
 ├─ Acoustic Features
 ├─ Linguistic Features
 └─ Temporal Features
      │
      ▼
Feature Engineering
      │
      ▼
Model Training
 ├─ Logistic Regression
 ├─ SVM
 ├─ Random Forest
 └─ Vision Transformer
      │
      ▼
Cross Validation
      │
      ▼
Statistical Testing
(Friedman Test)
```

---

# 🗂 Dataset

This project uses the **ADReSS Challenge 2020 dataset**.

Dataset characteristics:

| Property | Value |
|--------|------|
| Participants | 108 |
| Dementia patients | 54 |
| Healthy controls | 54 |
| Task | Cookie Theft picture description |
| Modality | Audio + transcripts |
| Language | English |

The dataset is **balanced for age and gender** to remove confounding variables.

---

# 🔬 Feature Engineering

## Acoustic Features

Extracted directly from audio signals.

Examples:

- MFCC (Mel Frequency Cepstral Coefficients)
- Spectral centroid
- Spectral bandwidth
- Pitch statistics
- Energy distribution

Libraries used:

```
librosa
numpy
scipy
```

These features capture **prosodic changes and articulation degradation**.

---

## Linguistic Features

Extracted from speech transcripts.

Examples:

- lexical diversity
- word repetition
- sentence complexity
- syntactic structure
- semantic richness

Libraries:

```
nltk
spacy
scikit-learn
```

These features capture **language simplification patterns** associated with dementia.

---

## Temporal Speech Biomarkers

Measures timing patterns in speech.

Examples:

- speech rate
- pause duration
- pause frequency
- hesitation markers
- disfluency rate

These features capture **cognitive processing delays**.

---

# 🤖 Machine Learning Models

| Model | Category |
|------|------|
| Logistic Regression | Linear baseline |
| Support Vector Machine | Kernel classifier |
| Random Forest | Ensemble tree model |
| Vision Transformer | Deep learning |

The **Vision Transformer (ViT)** learns spatial patterns from **Mel‑spectrogram representations of speech**.

---

# 🧪 Experimental Setup

Evaluation methodology:

- **5‑Fold Cross Validation**
- Stratified sampling
- Balanced dataset

Evaluation metrics:

- Accuracy
- Precision
- Recall
- F1 Score

---

# 📊 Experimental Results

## Cross‑Validation Results

| Fold | Logistic Regression | SVM | Random Forest |
|----|----|----|----|
| Fold1 | 0.63 | 0.63 | 0.70 |
| Fold2 | 0.59 | 0.59 | 0.67 |
| Fold3 | 0.56 | 0.56 | 0.63 |
| Fold4 | 0.62 | 0.62 | 0.69 |
| Fold5 | 0.61 | 0.61 | 0.69 |

---

## Average Accuracy

| Model | Mean Accuracy |
|------|--------------|
| Logistic Regression | 0.601 |
| SVM | 0.601 |
| Random Forest | **0.676** |

Random Forest achieved the **highest classification performance**.

---

# 📈 Statistical Model Comparison

To evaluate whether model performance differences are statistically meaningful, the **Friedman test** is applied.

Example output:

```
Friedman statistic = 6.4
p-value = 0.041
```

Interpretation:

- p < 0.05
- Performance differences are **statistically significant**.

Post‑hoc tests such as **Nemenyi test** may be used for pairwise comparisons.

---

# 🧮 MMSE Score Prediction

Beyond classification, the project explores predicting **MMSE cognitive scores**.

Regression models:

- Linear Regression
- Support Vector Regression
- Gradient Boosting Regressor

Metrics:

- RMSE
- MAE
- R²

This analysis explores whether **speech features track continuous cognitive decline**.

---

# ⚙️ Installation

Install dependencies:

```
pip install numpy
pip install pandas
pip install scikit-learn
pip install librosa
pip install matplotlib
pip install seaborn
pip install torch
pip install transformers
```

---

# ▶️ Running the Experiment

Clone repository

```
git clone https://github.com/your-repository/dementia-speech-ai
```

Move to directory

```
cd dementia-speech-ai
```

Launch notebook

```
jupyter notebook
```

Open

```
ADReSS_Dementia_Detection_friedman_vit_fixed.ipynb
```

---

# 🚀 Potential Applications

- Early dementia screening
- Remote cognitive monitoring
- Speech‑based digital biomarkers
- AI healthcare assistants
- Neurological disease research

---

# ⚠️ Disclaimer

This project is intended **for research purposes only**.

It does **not provide medical diagnosis** and should not replace clinical evaluation by healthcare professionals.

---

# 📚 References

ADReSS Challenge 2020

Luz et al. *Alzheimer’s Dementia Recognition through Spontaneous Speech*

WHO Global Status Report on Dementia

DementiaBank Pitt Corpus

---

# 📖 Citation

If you use this repository in your research, please cite:

```
@article{speech_dementia_detection_ai,
  title={Speech-based Dementia Detection using Multimodal AI},
  author={Research Project},
  year={2026}
}
```

---

# 👨‍🔬 Research Areas

Artificial Intelligence for Healthcare  
Speech Signal Processing  
Natural Language Processing  
Machine Learning for Neurological Disorders

