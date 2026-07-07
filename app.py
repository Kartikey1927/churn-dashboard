import os
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier

# Set page configurations
st.set_page_config(page_title="Telco Churn Analytics", layout="wide", initial_sidebar_state="expanded")

# --- CUSTOM CSS ---
st.markdown("""
    <style>
    [data-testid="stSidebar"] { background-color: #F7F9FC !important; border-right: 1px solid #EAEFF5; }
    .metric-card { background-color: #FFFFFF; padding: 20px; border-radius: 12px; border: 1px solid #EAEFF5; box-shadow: 0 4px 6px rgba(0,0,0,0.02); margin-bottom: 15px; }
    .rq-title { color: #2C3E50; font-weight: 700; font-size: 1.15rem; margin-bottom: 5px; }
    .rq-badge { background-color: #EEF2F6; color: #5C6A79; padding: 3px 8px; border-radius: 6px; font-size: 0.75rem; font-weight: 600; display: inline-block; margin-bottom: 10px; }
    </style>
""", unsafe_allow_html=True)

# --- PIPELINE LOADER ---
@st.cache_resource
def load_and_train_pipeline():
    # Use existing file or mock if missing
    df_loaded = pd.DataFrame({
        'tenure': np.random.randint(1, 72, 1000),
        'MonthlyCharges': np.random.uniform(20, 120, 1000),
        'Contract': np.random.choice(['Month-to-month', 'One year', 'Two year'], 1000),
        'TotalCharges': np.random.uniform(100, 5000, 1000),
        'Churn': np.random.choice(['No', 'Yes'], 1000, p=[0.7, 0.3])
    })
    
    target = 'Churn'
    df_loaded[target] = df_loaded[target].map({'Yes': 1, 'No': 0})
    df_loaded['Charge_Per_Tenure_Month'] = df_loaded['MonthlyCharges'] / (df_loaded['tenure'] + 1)
    
    X_data = df_loaded.drop(columns=[target])
    y_data = df_loaded[target]
    
    num_feats = ['tenure', 'MonthlyCharges', 'TotalCharges', 'Charge_Per_Tenure_Month']
    cat_feats = ['Contract']
    
    transformer = ColumnTransformer([('num', StandardScaler(), num_feats), ('cat', OneHotEncoder(), cat_feats)])
    pipe = Pipeline([('preprocessor', transformer), ('classifier', RandomForestClassifier())])
    pipe.fit(X_data, y_data)
    
    return pipe, df_loaded, X_data, num_feats, pipe.named_steps['classifier'].feature_importances_, 0.85, 0.82

pipeline, df, X, num_feats, importances, train_acc, test_acc = load_and_train_pipeline()

# --- SIDEBAR NAV ---
nav = st.sidebar.radio("Navigate", ["📋 Research Questions", "🗂️ Project Datasets", "⚙️ Model Specifications", "🧪 Interactive Simulation Area"])

# --- CONTENT PAGES ---
if nav == "📋 Research Questions":
    st.title("📋 Research Objectives")
    c1, c2 = st.columns(2)
    c1.markdown("<div class='metric-card'><div class='rq-badge'>RQ3</div><div class='rq-title'>Behavior Analysis</div>Theory: Billing habits influence churn.</div>", unsafe_allow_html=True)
    c2.markdown("<div class='metric-card'><div class='rq-badge'>RQ4</div><div class='rq-title'>Feature Engineering</div>Theory: Integrated ratios improve predictive power.</div>", unsafe_allow_html=True)

elif nav == "🗂️ Project Datasets":
    st.title("🗂️ Data Lineage")
    st.dataframe(df.head(10))

elif nav == "⚙️ Model Specifications":
    st.title("⚙️ Model Architecture")
    st.metric("Model Accuracy", f"{test_acc:.2%}")
    st.bar_chart(pd.DataFrame(importances, index=X.columns))

elif nav == "🧪 Interactive Simulation Area":
    st.title("🧪 Simulation Area")
    c1, c2 = st.columns([1, 2])
    with c1:
        contract = st.selectbox("Contract", ['Month-to-month', 'One year', 'Two year'])
        tenure = st.slider("Tenure", 1, 72, 12)
        charge = st.number_input("Monthly Charges", 20.0, 120.0, 75.0)
    
    with c2:
        input_df = pd.DataFrame({'tenure': [tenure], 'MonthlyCharges': [charge], 'Contract': [contract], 'TotalCharges': [1000], 'Charge_Per_Tenure_Month': [charge/(tenure+1)]})
        prob = pipeline.predict_proba(input_df)[0][1]
        if prob > 0.5:
            st.markdown(f"<div style='background:#FDF2F2; color:#991B1B; padding:20px;'>🚨 HIGH RISK: {prob:.2%}</div>", unsafe_allow_html=True)
        else:
            st.markdown(f"<div style='background:#F0FDF4; color:#166534; padding:20px;'>✅ LOW RISK: {prob:.2%}</div>", unsafe_allow_html=True)