# 🧠 Detecting Dementia using Natural Language Processing

This project explores the use of Natural Language Processing (NLP) and machine learning to detect dementia from transcribed speech. Using linguistic and cognitive features extracted from the ADReSS dataset, various classifiers are trained and evaluated to distinguish between dementia and control cases.

---

## 📊 Dataset

- **Source**: [DementiaBank ADReSS Challenge (2020)](https://dementia.talkbank.org/ADReSS-2020/)
- **Training Set**: 108 samples (54 control, 54 dementia)
- **Test Set**: 48 samples (labels used for final evaluation only)
- **Features Used**:
  - Speech transcript (text)
  - MMSE score
  - Age
  - Gender
  - Transcript length (derived)

---

## ⚙️ Preprocessing

- Text cleaning (punctuation, tags, disfluencies)
- Lowercasing and lemmatization
- Optional stopword removal (retained for deep learning models)
- MMSE imputation (class-wise median)
- Feature engineering: transcript length, gender encoding

---

## 🧠 Models and Features

### Text Representations:
- `TF-IDF` (1000-dimensional)
- `fastText` (averaged 300d vectors; trimmed vocab)
- `BERT` (768d [CLS] token from `bert-base-uncased`)

### Classifiers:
- Logistic Regression
- Support Vector Machine (SVM)
- Decision Tree
- Random Forest

---

## 🧪 Evaluation Metrics

- Accuracy  
- Precision  
- Recall  
- F1 Score  
- ROC AUC  
- Confusion Matrices  
- ROC Curves

---

## 📈 Results Summary

- **Best validation performance**: Decision Tree (TF-IDF / fastText / BERT) – F1 Score: 0.95
- **Best test performance**: Logistic Regression (BERT) and Decision Tree (fastText) – F1 Score: 0.91
- Embedding-based models generalize well; numeric features like MMSE significantly boost performance.

---

## 🛠️ Requirements

Install dependencies using:

```bash
pip install -r requirements.txt
