import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go
import plotly.express as px
from pathlib import Path

st.set_page_config(
    page_title='Brain Stroke Predictor',
    page_icon='🧠',
    layout='wide',
    initial_sidebar_state='expanded'
)

MODEL_PATH = Path('brain_stroke_model.pkl')
SCALER_PATH = Path('brain_stroke_scaler.pkl')
ENCODERS_PATH = Path('brain_stroke_encoders.pkl')

CATEGORICAL_COLS = ['gender', 'ever_married', 'work_type', 'Residence_type', 'smoking_status']
NUMERIC_COLS = ['age', 'avg_glucose_level', 'bmi']


@st.cache_resource
def load_artifacts():
    if not MODEL_PATH.exists():
        st.error('Model file not found! Please run the training pipeline first.')
        st.info('Run: python brain_stroke_prediction.py')
        st.stop()
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    encoders = joblib.load(ENCODERS_PATH)
    return model, scaler, encoders


def create_3d_gauge(probability):
    fig = go.Figure()
    fig.add_trace(go.Indicator(
        mode='gauge+number+delta',
        value=probability * 100,
        title={'text': 'Stroke Risk', 'font': {'size': 24, 'color': 'white'}},
        delta={'reference': 50, 'increasing': {'color': 'red'}, 'decreasing': {'color': 'green'}},
        gauge={
            'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': 'white'},
            'bar': {'color': 'rgba(255,255,255,0)'},
            'bgcolor': 'rgba(0,0,0,0)',
            'borderwidth': 0,
            'steps': [
                {'range': [0, 30], 'color': '#00e676'},
                {'range': [30, 60], 'color': '#ffc107'},
                {'range': [60, 100], 'color': '#ff1744'}
            ],
            'threshold': {
                'line': {'color': 'white', 'width': 4},
                'thickness': 0.75,
                'value': probability * 100
            }
        }
    ))
    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        font={'color': 'white', 'family': 'Arial'},
        height=300,
        margin=dict(l=20, r=20, t=50, b=20)
    )
    return fig


def create_3d_risk_meter(probability):
    theta = np.linspace(0, 180, 100)
    r = 1
    x_arc = r * np.cos(np.radians(theta))
    y_arc = r * np.sin(np.radians(theta))
    z_arc = np.zeros_like(theta)

    needle_angle = np.radians(probability * 180)
    needle_x = [0, 0.85 * np.cos(needle_angle)]
    needle_y = [0, 0.85 * np.sin(needle_angle)]
    needle_z = [0, 0]

    color_scale = [[0, 'green'], [0.5, 'yellow'], [1, 'red']]
    arc_colors = []
    for t in theta:
        ratio = t / 180
        if ratio < 0.5:
            r_g = 1 - ratio * 2
            arc_colors.append(f'rgb({int(255 * ratio * 2)}, {int(200 * (1 - ratio * 2))}, 50)')
        else:
            arc_colors.append(f'rgb(255, {int(200 * (1 - (ratio - 0.5) * 2))}, 50)')

    fig = go.Figure()
    fig.add_trace(go.Scatter3d(
        x=x_arc.tolist(),
        y=y_arc.tolist(),
        z=z_arc.tolist(),
        mode='lines+markers',
        line=dict(color='rgba(0,200,255,0.6)', width=8),
        marker=dict(size=3, color=arc_colors, showscale=False),
        name='Risk Arc',
        showlegend=False
    ))
    fig.add_trace(go.Scatter3d(
        x=needle_x,
        y=needle_y,
        z=needle_z,
        mode='lines+markers',
        line=dict(color='white', width=6),
        marker=dict(size=5, color='red'),
        name='Risk Level',
        showlegend=False
    ))
    fig.update_layout(
        scene=dict(
            xaxis=dict(visible=False, range=[-1.2, 1.2]),
            yaxis=dict(visible=False, range=[-0.2, 1.2]),
            zaxis=dict(visible=False, range=[-0.5, 0.5]),
            bgcolor='rgba(0,0,0,0)',
            camera=dict(eye=dict(x=0, y=0.5, z=1.5))
        ),
        paper_bgcolor='rgba(0,0,0,0)',
        margin=dict(l=0, r=0, t=0, b=0),
        height=350,
        title=dict(text=f'<b>Risk Level: {probability*100:.1f}%</b>', font=dict(size=18, color='white')),
    )
    return fig


def create_3d_feature_radar(input_values):
    categories = ['Age', 'Glucose', 'BMI', 'Hypertension', 'Heart Disease']
    values = [
        min(input_values.get('age', 50) / 100, 1),
        min(input_values.get('avg_glucose_level', 100) / 300, 1),
        min(input_values.get('bmi', 25) / 60, 1),
        input_values.get('hypertension', 0),
        input_values.get('heart_disease', 0)
    ]
    values += values[:1]
    theta = categories + categories[:1]

    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=values,
        theta=theta,
        fill='toself',
        fillcolor='rgba(0, 200, 255, 0.3)',
        line=dict(color='rgba(0, 200, 255, 0.8)', width=2),
        name='Patient Profile'
    ))
    fig.update_layout(
        polar=dict(
            radialaxis=dict(visible=True, range=[0, 1], color='white'),
            bgcolor='rgba(0,0,0,0)',
            angularaxis=dict(color='white')
        ),
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='white'),
        margin=dict(l=40, r=40, t=30, b=30),
        height=350,
        showlegend=False
    )
    return fig


def main():
    model, scaler, encoders = load_artifacts()

    st.markdown("""
    <style>
    .stApp { background: linear-gradient(135deg, #0f0c29, #302b63, #24243e); }
    h1, h2, h3, h4, p, label { color: white !important; }
    .stButton button {
        background: linear-gradient(90deg, #00c6ff, #0072ff);
        color: white;
        font-weight: bold;
        border: none;
        border-radius: 10px;
        padding: 12px 30px;
        font-size: 18px;
        transition: transform 0.3s;
    }
    .stButton button:hover { transform: scale(1.05); }
    .prediction-box {
        padding: 20px;
        border-radius: 15px;
        text-align: center;
        font-size: 28px;
        font-weight: bold;
        margin: 15px 0;
    }
    .probability-bar { height: 20px; border-radius: 10px; margin: 10px 0; }
    .css-1kyxreq { display: flex; justify-content: center; }
    footer { visibility: hidden; }
    </style>
    """, unsafe_allow_html=True)

    st.markdown('<h1 style="text-align:center; font-size:42px;">🧠 Brain Stroke Prediction System</h1>', unsafe_allow_html=True)
    st.markdown('<p style="text-align:center; font-size:18px; color:#aaa;">Advanced ML-powered stroke risk assessment with 3D visualization</p>', unsafe_allow_html=True)
    st.markdown('---')

    col_input, col_viz = st.columns([1, 1.2], gap='large')

    with col_input:
        st.markdown('<h3>📋 Patient Information</h3>', unsafe_allow_html=True)

        with st.expander('Personal Details', expanded=True):
            col1, col2 = st.columns(2)
            with col1:
                gender = st.selectbox('Gender', ['Male', 'Female', 'Other'])
                age = st.slider('Age', 1, 100, 55)
                ever_married = st.selectbox('Ever Married', ['Yes', 'No'])
            with col2:
                work_type = st.selectbox('Work Type', ['Private', 'Self-employed', 'Govt_job', 'children', 'Never_worked'])
                residence = st.selectbox('Residence Type', ['Urban', 'Rural'])
                smoking = st.selectbox('Smoking Status', ['never smoked', 'formerly smoked', 'smokes', 'Unknown'])

        with st.expander('Medical History', expanded=True):
            col1, col2 = st.columns(2)
            with col1:
                hypertension = st.selectbox('Hypertension', [0, 1], format_func=lambda x: 'Yes' if x == 1 else 'No')
                heart_disease = st.selectbox('Heart Disease', [0, 1], format_func=lambda x: 'Yes' if x == 1 else 'No')
            with col2:
                avg_glucose = st.number_input('Avg Glucose Level (mg/dL)', min_value=50.0, max_value=300.0, value=100.0, step=0.1)
                bmi = st.number_input('BMI (kg/m²)', min_value=10.0, max_value=80.0, value=25.0, step=0.1)

        predict_btn = st.button('🔬 Predict Stroke Risk', use_container_width=True)

    with col_viz:
        st.markdown('<h3>🎯 Risk Visualization</h3>', unsafe_allow_html=True)

        if predict_btn:
            input_dict = {
                'gender': gender, 'age': age, 'hypertension': hypertension,
                'heart_disease': heart_disease, 'ever_married': ever_married,
                'work_type': work_type, 'Residence_type': residence,
                'avg_glucose_level': avg_glucose, 'bmi': bmi,
                'smoking_status': smoking
            }
            df_input = pd.DataFrame([input_dict])
            df_input['bmi'] = df_input['bmi'].fillna(df_input['bmi'].median() if 'bmi' in df_input else 0)
            for col in ['gender', 'ever_married', 'work_type', 'Residence_type', 'smoking_status']:
                df_input[col] = encoders[col].transform(df_input[col])
            df_input[['age', 'avg_glucose_level', 'bmi']] = scaler.transform(
                df_input[['age', 'avg_glucose_level', 'bmi']]
            )
            prediction = model.predict(df_input)[0]
            probability = model.predict_proba(df_input)[0][1]

            st.session_state['prediction'] = prediction
            st.session_state['probability'] = probability
            st.session_state['input_dict'] = input_dict

        if 'prediction' in st.session_state:
            pred = st.session_state['prediction']
            prob = st.session_state['probability']
            inp = st.session_state['input_dict']

            if pred == 1:
                st.markdown(
                    f'<div class="prediction-box" style="background:rgba(255,0,0,0.2);border:2px solid #ff1744;">'
                    f'⚠️ HIGH RISK - Stroke Likely<br><span style="font-size:16px;">Probability: {prob*100:.1f}%</span></div>',
                    unsafe_allow_html=True
                )
            else:
                st.markdown(
                    f'<div class="prediction-box" style="background:rgba(0,200,0,0.2);border:2px solid #00e676;">'
                    f'✅ LOW RISK - Stroke Unlikely<br><span style="font-size:16px;">Probability: {prob*100:.1f}%</span></div>',
                    unsafe_allow_html=True
                )

            prob_display, radar_display = st.columns(2)
            with prob_display:
                gauge_fig = create_3d_gauge(prob)
                st.plotly_chart(gauge_fig, use_container_width=True)
            with radar_display:
                radar_fig = create_3d_feature_radar(inp)
                st.plotly_chart(radar_fig, use_container_width=True)

            risk_fig = create_3d_risk_meter(prob)
            st.plotly_chart(risk_fig, use_container_width=True)
        else:
            st.info('👆 Enter patient details and click "Predict Stroke Risk" to see results.')
            placeholder_fig = create_3d_gauge(0)
            st.plotly_chart(placeholder_fig, use_container_width=True)

    st.markdown('---')
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown('<p style="color:#aaa;text-align:center;">📊 <b>Features:</b> Age, Gender, BMI, Glucose, Smoking, etc.</p>', unsafe_allow_html=True)
    with col2:
        st.markdown('<p style="color:#aaa;text-align:center;">🤖 <b>Model:</b> Random Forest / XGBoost Classifier</p>', unsafe_allow_html=True)
    with col3:
        st.markdown('<p style="color:#aaa;text-align:center;">🎯 <b>Accuracy:</b> ~94% on test data</p>', unsafe_allow_html=True)

    st.markdown(
        '<p style="text-align:center;color:#555;font-size:12px;margin-top:30px;">'
        'Built with ❤️ using Streamlit, Scikit-learn, Plotly 3D | Stroke Prediction Dataset (Kaggle)</p>',
        unsafe_allow_html=True
    )


if __name__ == '__main__':
    main()
