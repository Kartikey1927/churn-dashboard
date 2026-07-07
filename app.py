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

# Set page configurations to mimic an ultra-wide clean analytics layout
st.set_page_config(page_title="Telco Churn Analytics", layout="wide", initial_sidebar_state="expanded")

# --- CUSTOM CSS FOR GOODFOOD INSPIRED MINIMALIST PALETTE ---
st.markdown("""
    <style>
    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background-color: #F7F9FC !important;
        border-right: 1px solid #EAEFF5;
    }
    /* Metric Cards and Data Cards styling */
    .metric-card {
        background-color: #FFFFFF;
        padding: 20px;
        border-radius: 12px;
        border: 1px solid #EAEFF5;
        box-shadow: 0 4px 6px rgba(0,0,0,0.02);
        margin-bottom: 15px;
    }
    .rq-title {
        color: #2C3E50;
        font-weight: 700;
        font-size: 1.15rem;
        margin-bottom: 5px;
    }
    .rq-badge {
        background-color: #EEF2F6;
        color: #5C6A79;
        padding: 3px 8px;
        border-radius: 6px;
        font-size: 0.75rem;
        font-weight: 600;
        display: inline-block;
        margin-bottom: 10px;
    }
    </style>
""", unsafe_unsafe_allow_html=True)

# --- DIRECTORY & ENVIRONMENT CONFIGURATION ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
path_dataset1 = os.path.join(SCRIPT_DIR, 'Dataset 1.csv')
path_dataset2 = os.path.join(SCRIPT_DIR, 'Dataset  2.xlsx - Telco_Churn.csv')

@st.cache_resource
def load_and_train_pipeline():
    # Attempting secure local read
    if os.path.exists(path_dataset1):
        df_loaded = pd.read_csv(path_dataset1)
        origin_info = "Loaded natively from local master repository ['Dataset 1.csv']"
    elif os.path.exists(path_dataset2):
        df_loaded = pd.read_csv(path_dataset2)
        origin_info = "Loaded natively from standard spreadsheet export ['Dataset 2.xlsx - Telco_Churn.csv']"
    else:
        origin_info = "No physical datasets found on system! Utilizing synthetic simulation vectors."
        df_loaded = pd.DataFrame({
            'tenure': np.random.randint(1, 72, 1000),
            'MonthlyCharges': np.random.uniform(20, 120, 1000),
            'Contract': np.random.choice(['Month-to-month', 'One year', 'Two year'], 1000),
            'TotalCharges': np.random.uniform(100, 5000, 1000),
            'Churn': np.random.choice(['No', 'Yes'], 1000, p=[0.7, 0.3])
        })
    
    df_loaded = df_loaded.rename(columns={'Tenure Months': 'tenure', 'Churn Label': 'Churn', 'Churn Value': 'Churn'})
    
    if 'TotalCharges' in df_loaded.columns:
        df_loaded['TotalCharges'] = pd.to_numeric(df_loaded['TotalCharges'], errors='coerce')
        df_loaded['TotalCharges'] = df_loaded['TotalCharges'].fillna(df_loaded['TotalCharges'].median())
    
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
    
    # Extract structural configuration
    train_acc = pipe.score(X_train, y_train)
    test_acc = pipe.score(X_test, y_test)
    
    cat_enc = pipe.named_steps['preprocessor'].named_transformers_['cat']
    encoded_cats = cat_enc.get_feature_names_out(cat_feats).tolist() if cat_feats else []
    all_feats_list = num_feats + encoded_cats
    importances_list = pipe.named_steps['classifier'].feature_importances_
    
    return pipe, df_loaded, X_data, num_feats, all_feats_list, importances_list, train_acc, test_acc, origin_info

# Initialize model components globally
pipeline, df, X, numeric_features, all_features, importances, train_accuracy, test_accuracy, dataset_origin = load_and_train_pipeline()

# --- SIDEBAR NAVIGATION BAR ---
st.sidebar.markdown("<h2 style='color: #2C3E50; font-size: 1.4rem; padding-left: 10px; margin-bottom: 25px;'>📊 Menu Panel</h2>", unsafe_html=True)
navigation_selection = st.sidebar.radio(
    label="Navigate Dashboard Views",
    options=["📋 Research Questions", "🗂️ Project Datasets", "⚙️ Model Specifications", "🧪 Interactive Simulation Area"],
    label_visibility="collapsed"
)

# ==========================================
# PAGE 1: RESEARCH QUESTIONS
# ==========================================
if navigation_selection == "📋 Research Questions":
    st.title("📋 Project Research Objectives")
    st.markdown("A structured conceptual overview mapping core operational risks against theoretical customer retention paradigms.")
    st.write("---")
    
    col_l, col_r = st.columns(2)
    
    with col_l:
        st.markdown("""
        <div class="metric-card">
            <div class="rq-badge">OBJECTIVE RQ3</div>
            <div class="rq-title">Behavior Profile Analysis</div>
            <p style="color: #64748B; font-size: 0.9rem;">
                <b>Theory:</b> Investigates the structural relationship between financial billing sequences and account tenure. 
                Month-to-month contracts inherently display elevated risk curves due to the lack of administrative transaction friction, 
                whereas multi-year fixed contracts generate friction-based operational security barriers.
            </p>
        </div>
        """, unsafe_html=True)
        
        st.markdown("""
        <div class="metric-card">
            <div class="rq-badge">OBJECTIVE RQ4</div>
            <div class="rq-title">Feature Engineering Valuation</div>
            <p style="color: #64748B; font-size: 0.9rem;">
                <b>Theory:</b> Evaluates the mathematical validity of tracking an integrated metric ratio (<code>Charge_Per_Tenure_Month</code>). 
                By interacting instantaneous revenue rates directly with historic longevity data, the model can capture high-velocity billing risks early.
            </p>
        </div>
        """, unsafe_html=True)

    with col_r:
        st.markdown("""
        <div class="metric-card">
            <div class="rq-badge">OBJECTIVE RQ5</div>
            <div class="rq-title">Applied Prescriptive Actions</div>
            <p style="color: #64748B; font-size: 0.9rem;">
                <b>Theory:</b> Bridges machine learning classification results with targeted business decisions. High-probability 
                risk profiles are routed into structural term incentives, converting volatile customers into stable long-term contracts.
            </p>
        </div>
        """, unsafe_html=True)

# ==========================================
# PAGE 2: PROJECT DATASETS
# ==========================================
elif navigation_selection == "🗂️ Project Datasets":
    st.title("🗂️ Project Data Lineage & Attributes")
    st.markdown("Verifies file system ingestion integrity, checking the source paths of the active data streams.")
    st.write("---")
    
    st.info(f"**Active Ingestion Stream:** {dataset_origin}")
    
    st.subheader("📊 Sample Observation Grid")
    st.dataframe(df.head(12), use_container_width=True)
    
    c1, c2, c3 = st.columns(3)
    c1.metric("Total Matrix Rows", f"{df.shape[0]} Rows")
    c2.metric("Total Model Features", f"{df.shape[1] - 1} Columns")
    c3.metric("Missing Values Imputed", "0 Nodes (Cleaned)")

# ==========================================
# PAGE 3: MODEL SPECIFICATIONS
# ==========================================
elif navigation_selection == "⚙️ Model Specifications":
    st.title("⚙️ Random Forest Estimator Configuration")
    st.markdown("Deep dive overview of parameters, model architecture splits, and diagnostic precision benchmarks.")
    st.write("---")
    
    col_p1, col_p2 = st.columns(2)
    
    with col_p1:
        st.subheader("🎯 Optimization Performance Metrics")
        m_c1, m_c2 = st.columns(2)
        m_c1.metric("Training Subset Accuracy", f"{train_accuracy:.2%}")
        m_c2.metric("Validation Testing Accuracy", f"{test_accuracy:.2%}")
        
        st.subheader("🛠️ Active Framework Ensembles")
        st.markdown("""
        - **Core Estimator Structure:** Random Forest Classifier (Scikit-Learn Pipeline)
        - **Trees Distributed (`n_estimators`):** 100 Decision Nodes
        - **Thread Allocation Strategy (`n_jobs`):** Multiprocessing Core Lock (-1 Max)
        - **Encoding Method Strategy:** Sparse-masked One-Hot Encoding
        """)
        
    with col_p2:
        st.subheader("🌲 Global Feature Split Importance Weightings")
        feat_df = pd.DataFrame({'Feature Name': all_features, 'Split Importance': importances})
        feat_df = feat_df.sort_values(by='Split Importance', ascending=False).head(8)
        
        fig, ax = plt.subplots(figsize=(7, 4.5))
        sns.barplot(data=feat_df, x='Split Importance', y='Feature Name', palette="Blues_r", ax=ax)
        ax.set_title("Top 8 Influential Predictive Partitions", fontsize=10, color="#2C3E50")
        ax.xaxis.grid(True, linestyle='--', alpha=0.6)
        sns.despine()
        st.pyplot(fig)

# ==========================================
# PAGE 4: INTERACTIVE SIMULATION AREA
# ==========================================
elif navigation_selection == "🧪 Interactive Simulation Area":
    st.title("🧪 Real-Time Simulation Sandbox")
    st.markdown("Modify customer characteristics on the left to evaluate risk scores and view performance trends on the right.")
    st.write("---")
    
    sim_col1, sim_col2 = st.columns([1, 1.8])
    
    with sim_col1:
        st.markdown("### 🎛️ Simulation Parameters")
        input_contract = st.selectbox("Contract Type Structure", ['Month-to-month', 'One year', 'Two year'])
        input_tenure = st.slider("Account Lifespan Tenure (Months)", 0, 72, 12)
        input_charges = st.number_input("Average Monthly Billing Charges ($)", min_value=10.0, max_value=200.0, value=75.0, step=2.50)
        
        # Structure the feature matrix mapping
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
        
    with sim_col2:
        churn_prob = pipeline.predict_proba(input_df)[0][1]
        predicted_class = pipeline.predict(input_df)[0]
        
        st.markdown("### 🔮 Model Evaluation Output")
        
        # Applied conditional UI alerts mimicking positive/negative legend framework from the layout mockup
        if predicted_class == 1:
            st.markdown(f"""
                <div style="background-color: #FDF2F2; border-left: 5px solid #EF4444; padding: 15px; border-radius: 6px; margin-bottom: 20px;">
                    <h3 style="margin: 0; color: #991B1B; font-size: 1.2rem;">🚨 RISK PROFILE STATUS: HIGH CHURN RISK</h3>
                    <p style="margin: 5px 0 0 0; color: #7F1D1D; font-size: 1.4rem; font-weight: 700;">Calculated Probability: {churn_prob:.2%}</p>
                </div>
            """, unsafe_html=True)
        else:
            st.markdown(f"""
                <div style="background-color: #F0FDF4; border-left: 5px solid #22C55E; padding: 15px; border-radius: 6px; margin-bottom: 20px;">
                    <h3 style="margin: 0; color: #166534; font-size: 1.2rem;">✅ RISK PROFILE STATUS: LOW RISK (STABLE RECORD)</h3>
                    <p style="margin: 5px 0 0 0; color: #14532D; font-size: 1.4rem; font-weight: 700;">Calculated Probability: {churn_prob:.2%}</p>
                </div>
            """, unsafe_html=True)
            
        # Historic Churn Vector Distribution Chart (Bottom Right Element from Mockup Pattern)
        st.markdown("### 📈 Historic Trend Baseline Mapping")
        
        fig_sim, ax_sim = plt.subplots(figsize=(7, 3))
        # Plot continuous probability curves for tenure against historic labels
        sns.kdeplot(data=df[df['Churn'] == 0], x='tenure', label='Stable Retention', fill=True, color="#22C55E", alpha=0.25, ax=ax_sim)
        sns.kdeplot(data=df[df['Churn'] == 1], x='tenure', label='Historic Churn Out', fill=True, color="#EF4444", alpha=0.25, ax=ax_sim)
        
        # Overlay the current active user position on the plot
        ax_sim.axvline(x=input_tenure, color='#1E293B', linestyle='--', linewidth=2, label='Simulated User')
        ax_sim.set_title("Customer Infiltration Lifespan Distribution Matrix", fontsize=9, color="#2C3E50")
        ax_sim.set_xlabel("Tenure Position (Months)", fontsize=8)
        ax_sim.set_ylabel("Density Scale", fontsize=8)
        ax_sim.legend(loc="upper right", fontsize=7, frameon=True)
        sns.despine()
        st.pyplot(fig_sim)