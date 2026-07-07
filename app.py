import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score
from tabulate import tabulate

# 1. Page Configuration
st.set_page_config(page_title="Churn Analytics", layout="wide")

# 2. Data & Model Pipeline
@st.cache_resource
def get_data_and_model():
    df = pd.DataFrame({
        'tenure': np.random.randint(1, 72, 100),
        'MonthlyCharges': np.random.uniform(20, 120, 100),
        'Contract': np.random.choice(['Month-to-month', 'One year', 'Two year'], 100),
        'TotalCharges': np.random.uniform(100, 5000, 100),
        'Churn': np.random.randint(0, 2, 100)
    })
    X = df.drop(columns=['Churn'])
    y = df['Churn']
    pre = ColumnTransformer([('num', StandardScaler(), ['tenure', 'MonthlyCharges', 'TotalCharges']),
                             ('cat', OneHotEncoder(), ['Contract'])])
    model = Pipeline([('pre', pre), ('clf', RandomForestClassifier())])
    model.fit(X, y)
    return df, model, X

df, model, X = get_data_and_model()

# 3. Sidebar Navigation
nav = st.sidebar.radio("Navigation", ["📋 Research Questions", "🗂️ Datasets", "⚙️ Model", "🧪 Test Area"])

# 4. Page Logic
if nav == "📋 Research Questions":
    st.title("Research Objectives")
    st.markdown("""<div style="background:#F7F9FC; padding:20px; border-radius:10px; border-left:5px solid #000;">
        <h3 style="color:#000;">