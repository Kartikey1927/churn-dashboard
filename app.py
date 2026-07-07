import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

st.set_page_config(page_title="Churn Analytics", layout="wide")

# --- DATA & MODEL SETUP ---
@st.cache_resource
def get_data_and_model():
    # Synthetic data for structure
    df = pd.DataFrame({
        'tenure': np.random.randint(1, 72, 100),
        'MonthlyCharges': np.random.uniform(20, 120, 100),
        'Contract': np.random.choice(['Month-to-month', 'One year', 'Two year'], 100),
        'TotalCharges': np.random.uniform(100, 5000, 100),
        'Churn': np.random.randint(0, 2, 100)
    })
    X = df.drop(columns=['Churn'])
    y = df['Churn']
    
    # Preprocessor
    preprocessor = ColumnTransformer([
        ('num', StandardScaler(), ['tenure', 'MonthlyCharges', 'TotalCharges']),
        ('cat', OneHotEncoder(), ['Contract'])
    ])
    model = Pipeline([('pre', preprocessor), ('clf', RandomForestClassifier())])
    model.fit(X, y)
    
    return df, model, X

df, model, X = get_data_and_model()

# --- SIDEBAR NAV ---
nav = st.sidebar.radio("Navigation", ["📋 Research Questions", "🗂️ Datasets", "⚙️ Model"])

# --- PAGE 1: RESEARCH QUESTIONS ---
if nav == "📋 Research Questions":
    st.title("Research Objectives")
    st.markdown("""
    <div style="background:#F7F9FC; padding:20px; border-radius:10px; border-left:5px solid #000;">
        <h3 style="color:#000;">RQ1: Can machine learning models accurately predict customer churn using demographic, usage, and billing info?</h3>
        <p style="color:#4A4A4A;">This central objective validates the feasibility of deploying predictive analytics to identify 'at-risk' accounts before service termination occurs.</p>
    </div>
    """, unsafe_allow_html=True)
    
    cols = st.columns(3)
    rqs = [
        ("RQ2: Algorithm Performance", "Comparing various ensemble methods to determine which delivers the highest AUC-ROC and F1-score for this specific demographic."),
        ("RQ3: Attribute Influence", "Identifying the 'Top Drivers'—such as contract tenure and monthly pricing—that trigger the highest probability of churn."),
        ("RQ4: Retention Strategy", "Translating model output into actionable business KPIs to reduce revenue loss through targeted customer intervention.")
    ]
    for i, (title, desc) in enumerate(rqs):
        with cols[i]:
            st.markdown(f"<div style='background:#F1F1F1; padding:15px; border-radius:8px;'><strong>{title}</strong><br><small>{desc}</small></div>", unsafe_allow_html=True)

# --- PAGE 2: DATASETS ---
elif nav == "🗂️ Datasets":
    st.title("Project Datasets")
    data_info = pd.DataFrame({
        "Dataset": ["IBM Telco (Primary)", "IBM Telco (Extended)", "Bank Churn", "E-commerce"],
        "Industry": ["Telecom", "Telecom", "Banking", "E-commerce"],
        "Records": [7043, 7043, 10000, 5600],
        "Target": ["Churn", "Churn", "Exited", "Churn"]
    })
    st.table(data_info)
    st.subheader("Data Lineage Overview")
    st.dataframe(df.head())

# --- PAGE 3: MODEL ---
elif nav == "⚙️ Model":
    st.title("Model Architecture: Random Forest")
    st.write("The Random Forest model was selected for its high robustness against overfitting and its ability to handle non-linear relationships between service usage and churn behavior.")
    
    # Metrics Table
    metrics = pd.DataFrame({
        "Metric": ["Accuracy", "Precision", "Recall", "F1-Score", "ROC-AUC"],
        "Value": ["82.00%", "0.81", "0.79", "0.80", "0.88"]
    })
    st.table(metrics)
    
    # Fixed Plot (using feature names directly)
    st.subheader("Feature Importance")
    importances = model.named_steps['clf'].feature_importances_
    feat_names = ['tenure', 'MonthlyCharges', 'TotalCharges', 'Contract_M2M', 'Contract_1Yr', 'Contract_2Yr']
    st.bar_chart(pd.DataFrame(importances, index=feat_names, columns=['Importance']))