import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import warnings
import plotly.express as px
import plotly.graph_objects as go

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score, roc_curve
)
from imblearn.over_sampling import SMOTE

try:
    from xgboost import XGBClassifier
    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False

warnings.filterwarnings('ignore')
sns.set_style('darkgrid')

DATA_PATH = 'stroke_data.csv'
MODEL_PATH = 'brain_stroke_model.pkl'
SCALER_PATH = 'brain_stroke_scaler.pkl'
ENCODERS_PATH = 'brain_stroke_encoders.pkl'

FEATURES = ['gender', 'age', 'hypertension', 'heart_disease', 'ever_married',
            'work_type', 'Residence_type', 'avg_glucose_level', 'bmi', 'smoking_status']
TARGET = 'stroke'

NUMERIC_COLS = ['age', 'avg_glucose_level', 'bmi']
CATEGORICAL_COLS = ['gender', 'ever_married', 'work_type', 'Residence_type', 'smoking_status']


def load_data(path=DATA_PATH):
    df = pd.read_csv(path)
    return df


def preprocess(df, fit_encoders=True, encoders=None, scaler=None):
    data = df.drop('id', axis=1, errors='ignore').copy()
    data['bmi'] = data['bmi'].fillna(data['bmi'].median())
    data['smoking_status'] = data['smoking_status'].replace('Unknown', data['smoking_status'].mode()[0])

    if fit_encoders:
        encoders = {}
        for col in CATEGORICAL_COLS:
            le = LabelEncoder()
            data[col] = le.fit_transform(data[col])
            encoders[col] = le
        scaler = StandardScaler()
        data[NUMERIC_COLS] = scaler.fit_transform(data[NUMERIC_COLS])
        return data, encoders, scaler
    else:
        for col in CATEGORICAL_COLS:
            data[col] = encoders[col].transform(data[col])
        data[NUMERIC_COLS] = scaler.transform(data[NUMERIC_COLS])
        return data


def train_models(X_train, y_train):
    models = {
        'Logistic Regression': LogisticRegression(class_weight='balanced', max_iter=1000, random_state=42),
        'Random Forest': RandomForestClassifier(class_weight='balanced', n_estimators=200, random_state=42),
    }
    if XGB_AVAILABLE:
        neg_ratio = (y_train == 0).sum() / (y_train == 1).sum()
        models['XGBoost'] = XGBClassifier(scale_pos_weight=neg_ratio, eval_metric='logloss', random_state=42, use_label_encoder=False)

    trained = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        trained[name] = model
    return trained


def evaluate_models(models, X_test, y_test):
    results = []
    for name, model in models.items():
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]
        results.append({
            'Model': name,
            'Accuracy': accuracy_score(y_test, y_pred),
            'Precision': precision_score(y_test, y_pred, zero_division=0),
            'Recall': recall_score(y_test, y_pred),
            'F1 Score': f1_score(y_test, y_pred, zero_division=0),
            'ROC AUC': roc_auc_score(y_test, y_prob),
        })
    return pd.DataFrame(results).sort_values('ROC AUC', ascending=False)


def plot_confusion_matrices(models, X_test, y_test, figsize=(15, 5)):
    fig, axes = plt.subplots(1, len(models), figsize=figsize)
    if len(models) == 1:
        axes = [axes]
    for ax, (name, model) in zip(axes, models.items()):
        cm = confusion_matrix(y_test, model.predict(X_test))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax, cbar=False)
        ax.set_title(f'{name}')
        ax.set_xlabel('Predicted')
        ax.set_ylabel('Actual')
    plt.tight_layout()
    return fig


def plot_roc_curves(models, X_test, y_test, figsize=(10, 8)):
    plt.figure(figsize=figsize)
    for name, model in models.items():
        y_prob = model.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        auc = roc_auc_score(y_test, y_prob)
        plt.plot(fpr, tpr, label=f'{name} (AUC = {auc:.3f})', lw=2)
    plt.plot([0, 1], [0, 1], 'k--', lw=1, label='Random')
    plt.xlim([0, 1])
    plt.ylim([0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves Comparison')
    plt.legend(loc='lower right')
    plt.grid(True, alpha=0.3)
    return plt.gcf()


def plot_feature_importance(model, feature_names, figsize=(10, 8)):
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
    elif hasattr(model, 'coef_'):
        importances = np.abs(model.coef_[0])
    else:
        return None
    indices = np.argsort(importances)
    plt.figure(figsize=figsize)
    plt.barh(range(len(indices)), importances[indices], align='center')
    plt.yticks(range(len(indices)), [feature_names[i] for i in indices])
    plt.xlabel('Importance')
    plt.title('Feature Importance')
    plt.tight_layout()
    return plt.gcf()


def save_artifacts(model, scaler, encoders, model_path=MODEL_PATH, scaler_path=SCALER_PATH, encoders_path=ENCODERS_PATH):
    joblib.dump(model, model_path)
    joblib.dump(scaler, scaler_path)
    joblib.dump(encoders, encoders_path)
    print(f'Model saved to {model_path}')
    print(f'Scaler saved to {scaler_path}')
    print(f'Encoders saved to {encoders_path}')


def load_artifacts(model_path=MODEL_PATH, scaler_path=SCALER_PATH, encoders_path=ENCODERS_PATH):
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    encoders = joblib.load(encoders_path)
    return model, scaler, encoders


def predict_single(model, scaler, encoders, input_dict):
    df_input = pd.DataFrame([input_dict])
    df_input['bmi'] = df_input['bmi'].fillna(df_input['bmi'].median() if 'bmi' in df_input else 0)
    for col in CATEGORICAL_COLS:
        if col in df_input.columns:
            df_input[col] = encoders[col].transform(df_input[col])
    df_input[NUMERIC_COLS] = scaler.transform(df_input[NUMERIC_COLS])
    pred = model.predict(df_input)[0]
    prob = model.predict_proba(df_input)[0][1]
    return int(pred), float(prob)


def run_pipeline():
    print('=' * 60)
    print('BRAIN STROKE PREDICTION PIPELINE')
    print('=' * 60)

    print('\n[1] Loading data...')
    df = load_data()
    print(f'Dataset shape: {df.shape}')
    print(f'Stroke cases: {df["stroke"].sum()} / {len(df)} ({df["stroke"].mean()*100:.2f}%)')

    print('\n[2] Preprocessing...')
    data, encoders, scaler = preprocess(df, fit_encoders=True)
    print('Missing values handled, categories encoded, features scaled.')

    X = data.drop(TARGET, axis=1)
    y = data[TARGET]

    print('\n[3] Splitting with SMOTE...')
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    smote = SMOTE(random_state=42)
    X_train_res, y_train_res = smote.fit_resample(X_train, y_train)
    print(f'Before SMOTE: {y_train.value_counts().to_dict()}')
    print(f'After SMOTE: {pd.Series(y_train_res).value_counts().to_dict()}')

    print('\n[4] Training models...')
    models = train_models(X_train_res, y_train_res)
    print(f'Trained {len(models)} models: {list(models.keys())}')

    print('\n[5] Evaluating...')
    results_df = evaluate_models(models, X_test, y_test)
    print(results_df.to_string(index=False))

    best_model_name = results_df.iloc[0]['Model']
    best_model = models[best_model_name]
    print(f'\nBest model: {best_model_name}')

    print('\n[6] Saving artifacts...')
    save_artifacts(best_model, scaler, encoders)

    print('\n[7] Feature importance...')
    plot_feature_importance(best_model, X.columns.tolist())
    plt.savefig('feature_importance.png', dpi=150, bbox_inches='tight')
    print('Feature importance plot saved.')

    print('\n[8] Sample prediction...')
    sample = {
        'gender': 'Male', 'age': 67, 'hypertension': 0, 'heart_disease': 1,
        'ever_married': 'Yes', 'work_type': 'Private', 'Residence_type': 'Urban',
        'avg_glucose_level': 228.69, 'bmi': 36.6, 'smoking_status': 'formerly smoked'
    }
    pred, prob = predict_single(best_model, scaler, encoders, sample)
    print(f'Sample: {sample}')
    print(f'Prediction: {"Stroke" if pred == 1 else "No Stroke"} (Probability: {prob:.2%})')

    print('\n' + '=' * 60)
    print('PIPELINE COMPLETED SUCCESSFULLY')
    print('=' * 60)
    return best_model, scaler, encoders


if __name__ == '__main__':
    run_pipeline()
