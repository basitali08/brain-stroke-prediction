<div align="center">
  <img src="https://readme-typing-svg.herokuapp.com?font=Fira+Code&weight=700&size=30&duration=3000&pause=1000&color=00E5FF&center=true&vCenter=true&width=600&lines=%F0%9F%A7%A0+Brain+Stroke+Prediction;Machine+Learning+%2B+3D+Visualization;End-to-End+ML+Project+2026" alt="Typing SVG" />
</div>

<br>

<div align="center">
  <img src="https://img.shields.io/badge/Python-3.10%2B-3776AB?style=for-the-badge&logo=python&logoColor=white&labelColor=1a1a2e">
  <img src="https://img.shields.io/badge/Scikit--Learn-1.3%2B-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white&labelColor=1a1a2e">
  <img src="https://img.shields.io/badge/XGBoost-2.0%2B-00E5FF?style=for-the-badge&logo=xgboost&logoColor=white&labelColor=1a1a2e">
  <img src="https://img.shields.io/badge/Streamlit-1.28%2B-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white&labelColor=1a1a2e">
  <img src="https://img.shields.io/badge/Plotly-3D-3F4F75?style=for-the-badge&logo=plotly&logoColor=white&labelColor=1a1a2e">
  <img src="https://img.shields.io/badge/SMOTE-Imbalanced-FF6F00?style=for-the-badge&logo=databricks&logoColor=white&labelColor=1a1a2e">
</div>

<br>

<div align="center">
  <img src="https://github-profile-trophy.vercel.app/?username=basitali08&theme=darkhub&no-frame=true&no-bg=true&column=3&title=Stars,Followers,Commits" width="400">
</div>

---

## 📋 Table of Contents

- [🌟 Overview](#-overview)
- [📊 Dataset](#-dataset)
- [⚙️ Project Pipeline](#️-project-pipeline)
- [📈 Model Performance](#-model-performance)
- [🎯 Key Findings](#-key-findings)
- [🖥️ Streamlit Dashboard](#️-streamlit-dashboard)
- [🚀 Quick Start](#-quick-start)
- [📁 Project Structure](#-project-structure)
- [🛠️ Tech Stack](#️-tech-stack)
- [📄 License](#-license)

---

## 🌟 Overview

> **Predict stroke risk from patient health data with 82.5% ROC AUC using Machine Learning — deployed as an interactive 3D dashboard.**

Stroke is the **2nd leading cause of death** worldwide. This project builds a complete ML system that:

- ✅ Analyzes **5,110 patient records** to detect stroke patterns
- ✅ Handles **severe class imbalance** (only 5% stroke cases) using **SMOTE**
- ✅ Compares **Logistic Regression, Random Forest & XGBoost**
- ✅ Achieves **76% recall** — the most critical metric for medical screening
- ✅ Deploys a **3D interactive dashboard** with animated risk visualizations

<br>

<div align="center">
  <img src="https://img.shields.io/badge/ROC_AUC-82.5%25-success?style=flat-square&labelColor=1a1a2e&color=00e676">
  <img src="https://img.shields.io/badge/Recall-76%25-success?style=flat-square&labelColor=1a1a2e&color=00e676">
  <img src="https://img.shields.io/badge/Accuracy-89.5%25-success?style=flat-square&labelColor=1a1a2e&color=00e676">
  <img src="https://img.shields.io/badge/Dataset-5,110_records-blue?style=flat-square&labelColor=1a1a2e&color=00c6ff">
</div>

---

## 📊 Dataset

**Source:** [Stroke Prediction Dataset](https://www.kaggle.com/datasets/fedesoriano/stroke-prediction-dataset) (Kaggle)

| Feature | Type | Description |
|---------|------|-------------|
| `gender` | Categorical | Male / Female / Other |
| `age` | Numeric | 0.32 – 82 years |
| `hypertension` | Binary | 0 = No, 1 = Yes |
| `heart_disease` | Binary | 0 = No, 1 = Yes |
| `ever_married` | Categorical | Yes / No |
| `work_type` | Categorical | Private, Self-employed, Govt_job, children, Never_worked |
| `Residence_type` | Categorical | Urban / Rural |
| `avg_glucose_level` | Numeric | 55 – 271 mg/dL |
| `bmi` | Numeric | 10 – 72 kg/m² |
| `smoking_status` | Categorical | never smoked, formerly smoked, smokes, Unknown |
| `stroke` | **Target** | **0** = No Stroke, **1** = Stroke |

> ⚠️ **Challenge:** Only **4.87%** (249/5110) records are stroke cases — heavily imbalanced.

---

## ⚙️ Project Pipeline

```
┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐
│  📥 Load  │ → │  🔍 EDA  │ → │  🧹 Clean │ → │ ⚖️ SMOTE │ → │ 🤖 Train │ → │  📊 Eval  │ → │  🚀 Deploy│
│   Data    │    │   Viz    │    │  Missing  │    │ Balance  │    │ 3 Models │    │ Compare   │    │ Streamlit│
└──────────┘    └──────────┘    └──────────┘    └──────────┘    └──────────┘    └──────────┘    └──────────┘
```

**Step-by-step:**
1. **Data Loading** — Load CSV with 5110 patient records
2. **EDA** — Visualize distributions, correlations, class imbalance
3. **Preprocessing** — Handle missing BMI, encode categories, scale features
4. **SMOTE** — Oversample minority class (stroke) to balance dataset
5. **Model Training** — Train Logistic Regression, Random Forest, XGBoost
6. **Evaluation** — Compare accuracy, recall, precision, F1, ROC AUC
7. **Deployment** — Best model served via Streamlit with 3D Plotly visualizations

---

## 📈 Model Performance

| Model | Accuracy | Precision | Recall | F1 Score | ROC AUC |
|-------|----------|-----------|--------|----------|---------|
| **Logistic Regression** ✅ | **75.5%** | 13.8% | **76.0%** | **23.3%** | **82.5%** |
| Random Forest | 88.3% | 11.1% | 20.0% | 14.3% | 77.7% |
| XGBoost | 89.5% | 13.9% | 22.0% | 17.1% | 77.6% |

> 🏆 **Logistic Regression** was chosen as the best model because **recall (76%)** is the most critical metric in medical diagnosis — missing a stroke (false negative) is far worse than a false alarm.

---

## 🎯 Key Findings

<div align="center">
  <table>
    <tr>
      <td align="center">🔴</td>
      <td><b>Age</b> — Strongest predictor. Risk increases significantly after 50+</td>
    </tr>
    <tr>
      <td align="center">🟡</td>
      <td><b>Glucose Level</b> — High glucose strongly correlates with stroke</td>
    </tr>
    <tr>
      <td align="center">🔴</td>
      <td><b>Heart Disease</b> — History doubles the stroke risk</td>
    </tr>
    <tr>
      <td align="center">🟡</td>
      <td><b>Hypertension</b> — High BP is a leading cause of stroke</td>
    </tr>
    <tr>
      <td align="center">🟢</td>
      <td><b>BMI & Smoking</b> — Moderate but measurable impact</td>
    </tr>
  </table>
</div>

---

## 🖥️ Streamlit Dashboard

The application features **3D interactive visualizations** powered by Plotly:

<div align="center">

| Widget | Description |
|--------|-------------|
| 🎯 **3D Gauge Chart** | Real-time risk meter with green/yellow/red color zones |
| 🌐 **3D Risk Arc** | Animated floating arc with rotating needle |
| 📡 **3D Radar Chart** | Patient profile across 5 health dimensions |
| 🚨 **Risk Alerts** | Color-coded high/low risk notifications |

</div>

### Screenshots

```
┌─────────────────────────────────────────────────────────┐
│  🧠 Brain Stroke Prediction System                       │
│  ─────────────────────────────────────────────────────  │
│  ┌──────────────┐  ┌────────────────────────────────┐  │
│  │  Patient Info │  │   🎯 Risk Gauge                │  │
│  │  ─────────── │  │   ┌──────────────────────┐    │  │
│  │  Gender: [M] │  │   │    ╭─── 82.5% ───╮   │  │  │
│  │  Age: [65]   │  │   │   ╱               ╲   │  │  │
│  │  BMI: [28]   │  │   │  │    STROKE       │  │  │  │
│  │  Glucose:..  │  │   │   ╲               ╱   │  │  │
│  │  Smoking:..  │  │   │    ╰─────────────╯   │  │  │
│  │  [PREDICT]   │  │   └──────────────────────┘  │  │
│  └──────────────┘  └────────────────────────────────┘  │
│  ┌────────────────────────────────────────────────┐  │
│  │  🌐 3D Risk Arc           📡 Patient Radar     │  │
│  │  ┌────────────┐          ┌────────────┐       │  │
│  │  │   ╱╲       │          │   ⬡       │       │  │
│  │  │  ╱  ╲      │          │  ╱ ╲      │       │  │
│  │  │ ╱    ╲     │          │ │   │     │       │  │
│  │  │╱ needle╲    │          │  ╲ ╱      │       │  │
│  │  └────────────┘          └────────────┘       │  │
│  └────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────┘
```

---

## 🚀 Quick Start

### Prerequisites
```bash
pip install numpy pandas matplotlib seaborn scikit-learn xgboost imbalanced-learn joblib plotly streamlit
```

### 1️⃣ Train the Model
```bash
python brain_stroke_prediction.py
```
Runs full pipeline: loads data → preprocesses → SMOTE → trains 3 models → evaluates → saves best model.

### 2️⃣ Launch 3D Dashboard
```bash
streamlit run streamlit_app.py
```
Opens interactive dashboard in your browser at `http://localhost:8501`.

### 3️⃣ Explore the Notebook
Open `BrainStroke_Prediction.ipynb` in Jupyter Lab / VS Code for the complete step-by-step analysis with visualizations.

---

## 📁 Project Structure

```
📦 brain-stroke-prediction
├── 📄 BrainStroke_Prediction.ipynb   # Jupyter notebook (full analysis)
├── 📄 brain_stroke_prediction.py     # Python pipeline module
├── 📄 streamlit_app.py               # Streamlit 3D dashboard
├── 📄 stroke_data.csv                # Dataset (5110 records)
├── 📄 brain_stroke_model.pkl         # Trained Logistic Regression model
├── 📄 brain_stroke_scaler.pkl        # StandardScaler
├── 📄 brain_stroke_encoders.pkl      # LabelEncoders
├── 📄 feature_importance.png         # Feature importance plot
├── 📄 brain_stroke_prediction_seo_post.md  # SEO blog post
├── 📄 README.md                      # You are here 📍
└── 📄 .gitignore
```

---

## 🛠️ Tech Stack

<div align="center">

| Technology | Purpose |
|------------|---------|
| <img src="https://img.shields.io/badge/Python-3776AB?style=flat&logo=python&logoColor=white"> | Core language |
| <img src="https://img.shields.io/badge/Scikit--Learn-F7931E?style=flat&logo=scikit-learn&logoColor=white"> | ML models & preprocessing |
| <img src="https://img.shields.io/badge/XGBoost-00E5FF?style=flat&logo=xgboost&logoColor=white"> | Gradient boosting classifier |
| <img src="https://img.shields.io/badge/Imbalanced--Learn-FF6F00?style=flat&logo=databricks&logoColor=white"> | SMOTE oversampling |
| <img src="https://img.shields.io/badge/Streamlit-FF4B4B?style=flat&logo=streamlit&logoColor=white"> | Web app framework |
| <img src="https://img.shields.io/badge/Plotly-3F4F75?style=flat&logo=plotly&logoColor=white"> | 3D visualizations |
| <img src="https://img.shields.io/badge/Joblib-FF6F00?style=flat&logo=python&logoColor=white"> | Model serialization |
| <img src="https://img.shields.io/badge/Pandas-150458?style=flat&logo=pandas&logoColor=white"> | Data manipulation |
| <img src="https://img.shields.io/badge/Seaborn-3776AB?style=flat&logo=python&logoColor=white"> | Statistical plotting |

</div>

---

## 📄 License

This project is for **educational and research purposes**. Dataset used under Kaggle's community license.

---

<div align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=0:1a1a2e,50:00c6ff,100:0072ff&height=100&section=footer&text=⭐%20Star%20if%20you%20found%20this%20useful!%20⭐&fontSize=20&fontColor=white" alt="footer" />
</div>

<div align="center">
  
**Built with ❤️ using Python, Scikit-learn, XGBoost, SMOTE, Streamlit & Plotly 3D**
  
[![GitHub stars](https://img.shields.io/github/stars/basitali08/brain-stroke-prediction?style=social)](https://github.com/basitali08/brain-stroke-prediction)
[![GitHub forks](https://img.shields.io/github/forks/basitali08/brain-stroke-prediction?style=social)](https://github.com/basitali08/brain-stroke-prediction)

</div>
