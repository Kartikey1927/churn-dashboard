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

# Set layout 
st.set_page_config(page_title="Churn Dashboard", layout="wide")

st.title("📊 Customer Churn & Research Question Analytics Dashboard")
st.markdown("This interactive panel applies live values to your trained Random Forest pipeline to compute real-time churn predictions and evaluate project RQs.")

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
path_dataset1 = os.path.join(SCRIPT_DIR, 'Dataset 1.csv')
path_dataset2 = os.path.join(SCRIPT_DIR, 'Dataset  2.xlsx - Telco_Churn.csv')

# Use a pristine cache name to force Streamlit out of stale RAM states
@st.cache_resource
def final_clean_pipeline_loader():
    if os.path.exists(path_dataset1):
        df_loaded = pd.read_csv(path_dataset1)
    elif os.path.exists(path_dataset2):
        df_loaded = pd.read_csv(path_dataset2)
    else:
        st.error("No dataset file found! Generating mock background elements...")
        df_loaded = pd.DataFrame({
            'tenure': np.random.randint(1, 72, 1000),
            'MonthlyCharges': np.random.uniform(20, 120, 1000),
            'Contract': np.random.choice(['Month-to-month', 'One year', 'Two year'], 1000),
            'Churn': np.random.choice(['No', 'Yes'], 1000, p=[0.7, 0.3])
        })
    
    df_loaded = df_loaded.rename(columns={'Tenure Months': 'tenure', 'Churn Label': 'Churn', 'Churn Value': 'Churn'})
    
    # EXPLICIT MULTI-STEP CONVERSION: Prevents inline reduction errors on strings
    if 'TotalCharges' in df_loaded.columns:
        df_loaded['TotalCharges'] = pd.to_numeric(df_loaded['TotalCharges'], errors='coerce')
        calculated_median = df_loaded['TotalCharges'].median()
        df_loaded['TotalCharges'] = df_loaded['TotalCharges'].fillna(calculated_median)
    
    for col in ['customerID', 'CustomerID', 'Count', 'Country', 'State', 'City', 'Zip Code', 'Lat Long', 'Latitude', 'Longitude', 'Churn Score', 'CLTV', 'Churn Reason']:
        if col in df_loaded.columns:
            df_loaded = df_loaded.drop(columns=[col])
            
    target = 'Churn'
    df_loaded[target] = df_loaded[target].astype(str).str.strip().str.lower().map({'yes': 1, '1': 1, 'true': 1, 'no': 0, '0': 0, 'false': 0}).fillna(0).astype(int)
    
    if 'MonthlyCharges' in df_loaded.columns and 'tenure' in df_loaded.columns:
        df_loaded['Charge_Per_Tenure_Month'] = df_loaded['MonthlyCharges'] / (df_loaded['tenure'] + 1)
        
    X_data = df_loaded.drop(columns=[target])
    y_data = df_loaded[target]
    
    num_feats = X_data.select_dtypes(include=['int64', 'float64']).columns.tolist()
    cat_feats = X_data.select_dtypes(include=['object', 'category', 'string']).columns.tolist()
    
    transformer = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), num_feats),
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), cat_feats)
        ]
    )
    
    pipe = Pipeline(steps=[
        ('preprocessor', transformer),
        ('classifier', RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1))
    ])
    
    X_train, X_test, y_train, y_test = train_test_split(X_data, y_data, test_size=0.2, random_state=42, stratify=y_data)
    pipe.fit(X_train, y_train)
    
    cat_enc = pipe.named_steps['preprocessor'].named_transformers_['cat']
    encoded_cats = cat_enc.get_feature_names_out(cat_feats).tolist() if cat_feats else []
    all_feats_list = num_feats + encoded_cats
    importances_list = pipe.named_steps['classifier'].feature_importances_
    
    return pipe, df_loaded, X_data, num_feats, all_feats_list, importances_list

pipeline, df, X, numeric_features, all_features, importances = final_clean_pipeline_loader()

# App Interface Columns
col1, col2 = st.columns([1, 2])

with col1:
    st.header("🎛️ Simulation Parameters")
    input_contract = st.selectbox("Contract Type", ['Month-to-month', 'One year', 'Two year'])
    input_tenure = st.slider("Customer Tenure (Months)", 0, 72, 12)
    input_charges = st.number_input("Monthly Charges ($)", min_value=10.0, max_value=3000.0, value=75.0, step=5.0)
    
    if input_charges > 150:
        st.warning(f"💡 Outlier Notice: ${input_charges:.2f} is significantly higher than historical baseline records (Max ~$120). Calculations might stretch into probability extremes.")

    sample_data = {}
    for col in X.columns:
        if col in numeric_features:
            sample_data[col] = [df[col].median()]
        else:
            sample_data[col] = [df[col].mode()[0] if not df[col].mode().empty else ""]
            
    sample_data['Contract'] = [input_contract]
    sample_data['tenure'] = [input_tenure]
    sample_data['MonthlyCharges'] = [input_charges]
    sample_data['Charge_Per_Tenure_Month'] = [input_charges / (input_tenure + 1)]
    input_df = pd.DataFrame(sample_data)

with col2:
    st.header("🔮 Model Target Evaluation")
    churn_prob = pipeline.predict_proba(input_df)[0][1]
    predicted_class = pipeline.predict(input_df)[0]
    
    m_col1, m_col2 = st.columns(2)
    if predicted_class == 1:
        m_col1.metric("Risk Assessment Profile", "🚨 HIGH RISK", "Action Flagged")
    else:
        m_col1.metric("Risk Assessment Profile", "✅ LOW RISK", "Stable Record", delta_color="inverse")
    m_col2.metric("Calculated Churn Probability", f"{churn_prob:.2%}")
    
    st.subheader("📋 Contextual Research Question Outputs")
    
    st.markdown(f"**[RQ3 Findings - Behavior Profile]:** This user maps to a `{input_contract}` billing sequence. "
                f"{'Month-to-month models lack structural inertia, significantly intensifying churn vulnerability.' if input_contract == 'Month-to-month' else 'Contract requirements act as an operational security barrier, retaining the profile cleanly.'} "
                f"Tenure account age position ({input_tenure} months) holds a global model relative splits weight of `{importances[numeric_features.index('tenure')]:.4f}`.")
    
    eng_val = input_charges / (input_tenure + 1)
    st.markdown(f"**[RQ4 Findings - Feature Engineering]:** Custom ratio interaction column `Charge_Per_Tenure_Month` evaluates to `{eng_val:.4f}`. "
                f"This engineered feature impacts **{importances[all_features.index('Charge_Per_Tenure_Month')]:.2%}** of total node partitioning metrics across the estimator system.")
    
    st.markdown("**[RQ5 Findings - Applied Prescriptive Action]:**")
    if predicted_class == 1:
        if input_contract == 'Month-to-month':
            st.error("⚠️ **Strategy:** Account metrics breach stable variance thresholds. Target instantly with structural conversion incentives to transition them into standard fixed terms.")
        else:
            st.error("⚠️ **Strategy:** Churn warning active despite structural contract length. Initiate usage diagnostic audits immediately.")
    else:
        st.success("✨ **Strategy:** Standard profile consistency validated. No proactive financial discount intervention necessary.")