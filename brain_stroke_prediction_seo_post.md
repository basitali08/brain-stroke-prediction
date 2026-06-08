# 🧠 Brain Stroke Prediction Using Machine Learning – Complete Project Guide

**Meta Description:** Learn how to build a brain stroke prediction system using Machine Learning with Python. This guide covers data preprocessing, SMOTE, model training (Logistic Regression, Random Forest, XGBoost), and a Streamlit 3D visualization dashboard. Includes full code and dataset.

**Keywords:** brain stroke prediction, stroke prediction machine learning, stroke dataset analysis, ML stroke detection, healthcare AI project, stroke risk prediction Python, SMOTE imbalanced data, Streamlit 3D visualization, Random Forest stroke, XGBoost classifier

**URL Slug:** `/brain-stroke-prediction-machine-learning`

---

## Table of Contents

1. [Introduction](#introduction)
2. [Dataset Overview](#dataset-overview)
3. [Project Workflow](#project-workflow)
4. [Exploratory Data Analysis](#exploratory-data-analysis)
5. [Data Preprocessing](#data-preprocessing)
6. [Handling Class Imbalance with SMOTE](#handling-class-imbalance-with-smote)
7. [Model Training & Evaluation](#model-training--evaluation)
8. [Results & Model Comparison](#results--model-comparison)
9. [Feature Importance Analysis](#feature-importance-analysis)
10. [Deployment with Streamlit & 3D Visualization](#deployment-with-streamlit--3d-visualization)
11. [How to Run This Project](#how-to-run-this-project)
12. [Conclusion & Future Work](#conclusion--future-work)

---

## Introduction

Stroke is one of the leading causes of death and long-term disability worldwide. According to the World Health Organization (WHO), 15 million people suffer from stroke annually, with 5 million dying and another 5 million permanently disabled. Early prediction of stroke risk can save lives through timely medical intervention.

This project builds an end-to-end **Brain Stroke Prediction System** using **Machine Learning** that:

- Analyzes patient health records to predict stroke likelihood
- Handles real-world challenges like **missing data** and **severe class imbalance**
- Compares multiple algorithms: **Logistic Regression, Random Forest, and XGBoost**
- Deploys an interactive **Streamlit dashboard with 3D visualizations** (Plotly)

The complete code is available in the project files: `brain_stroke_prediction.py` (pipeline), `BrainStroke_Prediction.ipynb` (Jupyter notebook), and `streamlit_app.py` (deployment).

---

## Dataset Overview

We use the **Stroke Prediction Dataset** from Kaggle (fedesoriano), containing **5,110 patient records** with 12 attributes:

| Column | Type | Description |
|---|---|---|
| `id` | int | Unique patient identifier |
| `gender` | categorical | Male / Female / Other |
| `age` | float | Patient age (0.32–82 years) |
| `hypertension` | binary | 0 = No, 1 = Yes |
| `heart_disease` | binary | 0 = No, 1 = Yes |
| `ever_married` | categorical | Yes / No |
| `work_type` | categorical | Private, Self-employed, Govt_job, children, Never_worked |
| `Residence_type` | categorical | Urban / Rural |
| `avg_glucose_level` | float | Average blood glucose (mg/dL) |
| `bmi` | float | Body Mass Index (kg/m²) |
| `smoking_status` | categorical | never smoked, formerly smoked, smokes, Unknown |
| `stroke` | binary | **Target:** 1 = Stroke, 0 = No Stroke |

**Critical finding:** Only **4.87% (249/5110)** records are stroke cases — a heavily **imbalanced dataset** that required special handling.

---

## Project Workflow

```
Data Loading → EDA → Preprocessing → SMOTE → Train/Test Split → 
Model Training (LogReg, RF, XGB) → Evaluation → 
Save Best Model → Streamlit Deployment with 3D Viz
```

---

## Exploratory Data Analysis

Key visual insights from the data:

### Stroke Distribution
Only ~5% of patients had a stroke, confirming severe class imbalance. This mirrors real-world medical data where positive cases are rare.

### Age & Stroke Risk
Stroke risk increases significantly after age 50. The 60–80 age bracket shows the highest concentration of stroke cases.

### Glucose & BMI
- Patients with stroke tend to have **higher avg_glucose_level** (median ~120 vs ~90 for non-stroke)
- **BMI** shows a weaker but visible trend — higher BMI correlates with slightly elevated risk

### Smoking & Work Type
- "Formerly smoked" and "smokes" categories show higher stroke proportions
- "Private" work type and "Urban" residents show slightly elevated risk

### Correlation Heatmap
Age has the strongest positive correlation with stroke, followed by avg_glucose_level and hypertension.

---

## Data Preprocessing

Steps applied to prepare data for modeling:

1. **Drop `id` column** — not a predictive feature
2. **Handle missing BMI values** — filled with median (BMI had ~4% missing)
3. **Handle Unknown smoking_status** — replaced with mode ("never smoked")
4. **Label Encode categorical variables** — gender, ever_married, work_type, Residence_type, smoking_status
5. **Standard Scale numeric features** — age, avg_glucose_level, bmi (StandardScaler)

---

## Handling Class Imbalance with SMOTE

Class imbalance (4.87% stroke vs 95.13% no-stroke) causes models to predict "No Stroke" for every case, achieving high accuracy but zero clinical value.

**Solution: SMOTE (Synthetic Minority Oversampling Technique)**

SMOTE creates synthetic stroke samples by interpolating between existing minority class instances. After SMOTE:

| Class | Before SMOTE | After SMOTE |
|---|---|---|
| No Stroke | 3,889 | 3,889 |
| Stroke | 199 | **3,889** |

This balanced dataset allowed models to learn meaningful stroke patterns.

---

## Model Training & Evaluation

Three classifiers were trained and compared:

### 1. Logistic Regression
- Class weight: `balanced`
- Max iterations: 1000
- **Strength:** Best recall (76%) and ROC AUC (82.5%) — most reliable for medical screening

### 2. Random Forest
- 200 estimators, class_weight: `balanced`
- **Strength:** High accuracy (88.3%), but low recall (20%)

### 3. XGBoost
- `scale_pos_weight` set to imbalance ratio
- **Strength:** Highest accuracy (89.5%), good generalization

---

## Results & Model Comparison

| Model | Accuracy | Precision | Recall | F1 Score | ROC AUC |
|---|---|---|---|---|---|
| **Logistic Regression** | **75.5%** | 13.8% | **76.0%** | **23.3%** | **82.5%** |
| Random Forest | 88.3% | 11.1% | 20.0% | 14.3% | 77.7% |
| XGBoost | 89.5% | 13.9% | 22.0% | 17.1% | 77.6% |

**Why Logistic Regression was chosen as the best model:**

In medical diagnosis, **recall (sensitivity)** matters more than accuracy. A false negative (missing a real stroke) is far worse than a false positive. Logistic Regression achieved **76% recall** — the highest among all models, meaning it correctly identifies 76% of actual stroke cases. Its **ROC AUC of 82.5%** also indicates strong discriminatory power.

---

## Feature Importance Analysis

Top factors influencing stroke prediction (Logistic Regression coefficients):

1. **Age** — strongest predictor; stroke risk increases significantly with age
2. **Avg Glucose Level** — high glucose is a major stroke risk factor
3. **Heart Disease** — history of heart disease doubles stroke risk
4. **Hypertension** — high blood pressure is a leading stroke cause
5. **BMI** — obesity contributes to elevated risk
6. **Ever Married** — correlated with age (older patients more likely married)
7. **Smoking Status** — tobacco use increases stroke risk
8. **Gender** — marginal effect; males slightly higher risk

---

## Deployment with Streamlit & 3D Visualization

The final model is deployed as a **Streamlit web application** featuring:

### 3D Interactive Visualizations
- **3D Gauge Chart** — Real-time risk meter (green/yellow/red zones)
- **3D Risk Arc** — Floating arc with animated needle showing stroke probability
- **3D Radar Chart** — Patient profile visualization across 5 health dimensions

### Features
- Sidebar input form for patient demographics and medical history
- Real-time prediction with probability display
- Color-coded risk alerts (green = low, red = high)
- Mobile-responsive design
- Dark theme with gradient background

---

## How to Run This Project

### Prerequisites
```bash
pip install numpy pandas matplotlib seaborn scikit-learn xgboost imbalanced-learn joblib plotly streamlit
```

### Step 1: Train the Model
```bash
python brain_stroke_prediction.py
```
This runs the full pipeline: loads data, preprocesses, trains models, evaluates, and saves artifacts.

### Step 2: Launch Streamlit Dashboard
```bash
streamlit run streamlit_app.py
```
Opens the interactive 3D prediction dashboard in your browser.

### Step 3: Explore the Notebook
Open `BrainStroke_Prediction.ipynb` in Jupyter Lab/VS Code for step-by-step analysis.

---

## Conclusion & Future Work

This project demonstrates a complete ML pipeline for stroke prediction with **82.5% ROC AUC** using Logistic Regression with SMOTE balancing. The Streamlit dashboard provides an intuitive, visually rich interface for real-time risk assessment.

### Key Takeaways
- Class imbalance is critical in medical ML — SMOTE significantly improves recall
- Logistic Regression with balanced weights outperformed ensemble methods on recall
- 3D visualization makes risk assessment intuitive for doctors and patients
- Age, glucose, and heart disease are the strongest stroke predictors

### Future Improvements
- Add **Deep Learning** (Neural Networks) for comparison
- Incorporate **CT/MRI imaging data** for multimodal prediction
- Deploy as a **cloud API** (FastAPI + Docker)
- Add **SHAP explainability** for model interpretability
- Collect more data to improve minority class representation

---

*Built with ❤️ using Python, Scikit-learn, XGBoost, SMOTE, Streamlit & Plotly 3D*

*Dataset: Stroke Prediction Dataset by fedesoriano (Kaggle)*
