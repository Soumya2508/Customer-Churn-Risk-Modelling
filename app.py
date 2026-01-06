"""
📊 Customer Churn Risk Modeling Dashboard
A comprehensive Streamlit application for analyzing and predicting customer churn.
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# ML libraries
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                             f1_score, roc_auc_score, confusion_matrix,
                             classification_report, roc_curve)
from imblearn.over_sampling import SMOTE

# Page Configuration
st.set_page_config(
    page_title="Customer Churn Analytics",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for premium design
st.markdown("""
<style>
    /* Main container styling */
    .main {
        background: linear-gradient(135deg, #0f0f23 0%, #1a1a2e 50%, #16213e 100%);
    }
    
    /* Metric cards */
    .metric-card {
        background: linear-gradient(145deg, #1e1e2f, #252540);
        border-radius: 16px;
        padding: 24px;
        border: 1px solid rgba(255, 255, 255, 0.1);
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
        text-align: center;
        transition: transform 0.3s ease, box-shadow 0.3s ease;
    }
    
    .metric-card:hover {
        transform: translateY(-4px);
        box-shadow: 0 12px 48px rgba(0, 0, 0, 0.4);
    }
    
    .metric-value {
        font-size: 2.5rem;
        font-weight: 700;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }
    
    .metric-label {
        font-size: 0.9rem;
        color: #a0a0b0;
        text-transform: uppercase;
        letter-spacing: 1.5px;
        margin-top: 8px;
    }
    
    /* Section headers */
    .section-header {
        font-size: 1.8rem;
        font-weight: 600;
        color: #ffffff;
        margin: 2rem 0 1.5rem 0;
        padding-bottom: 0.5rem;
        border-bottom: 3px solid;
        border-image: linear-gradient(90deg, #667eea, #764ba2, transparent) 1;
    }
    
    /* Status indicators */
    .status-high {
        background: linear-gradient(135deg, #ff6b6b, #ee5a5a);
        padding: 8px 16px;
        border-radius: 20px;
        color: white;
        font-weight: 600;
    }
    
    .status-medium {
        background: linear-gradient(135deg, #feca57, #ff9f43);
        padding: 8px 16px;
        border-radius: 20px;
        color: #333;
        font-weight: 600;
    }
    
    .status-low {
        background: linear-gradient(135deg, #1dd1a1, #10ac84);
        padding: 8px 16px;
        border-radius: 20px;
        color: white;
        font-weight: 600;
    }
    
    /* Cards container */
    .insight-card {
        background: rgba(255, 255, 255, 0.03);
        border-radius: 12px;
        padding: 20px;
        margin: 10px 0;
        border-left: 4px solid #667eea;
    }
    
    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1a1a2e 0%, #16213e 100%);
    }
    
    /* Prediction result */
    .prediction-result {
        font-size: 1.5rem;
        padding: 24px;
        border-radius: 16px;
        text-align: center;
        margin: 20px 0;
    }
    
    .churn-high {
        background: linear-gradient(145deg, #2d1a1a, #3d2020);
        border: 2px solid #ff6b6b;
        color: #ff8888;
    }
    
    .churn-low {
        background: linear-gradient(145deg, #1a2d1a, #203d20);
        border: 2px solid #1dd1a1;
        color: #44e0b0;
    }
    
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    
    /* Custom button */
    .stButton>button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        padding: 12px 32px;
        border-radius: 25px;
        font-weight: 600;
        letter-spacing: 0.5px;
        transition: all 0.3s ease;
    }
    
    .stButton>button:hover {
        transform: scale(1.05);
        box-shadow: 0 8px 25px rgba(102, 126, 234, 0.4);
    }
</style>
""", unsafe_allow_html=True)


@st.cache_data
def load_data():
    """Load and cache the dataset"""
    df = pd.read_csv('WA_Fn-UseC_-Telco-Customer-Churn.csv')
    return df


@st.cache_data
def preprocess_data(df):
    """Clean and preprocess the data"""
    df_clean = df.copy()
    
    # Drop customerID
    df_clean = df_clean.drop('customerID', axis=1)
    
    # Fix TotalCharges
    df_clean['TotalCharges'] = pd.to_numeric(df_clean['TotalCharges'], errors='coerce')
    
    # Drop rows with missing values
    df_clean = df_clean.dropna()
    
    return df_clean


@st.cache_resource
def train_models(df):
    """Train ML models with SMOTE"""
    df_encoded = df.copy()
    
    # Identify and encode categorical columns
    categorical_cols = df_encoded.select_dtypes(include=['object']).columns.tolist()
    categorical_cols.remove('Churn')
    
    label_encoders = {}
    for col in categorical_cols:
        le = LabelEncoder()
        df_encoded[col] = le.fit_transform(df_encoded[col])
        label_encoders[col] = le
    
    # Encode target
    df_encoded['Churn'] = (df_encoded['Churn'] == 'Yes').astype(int)
    
    # Split features and target
    X = df_encoded.drop('Churn', axis=1)
    y = df_encoded['Churn']
    
    # Apply SMOTE
    smote = SMOTE(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X, y)
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X_resampled, y_resampled, test_size=0.2, random_state=42, stratify=y_resampled
    )
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train Logistic Regression
    lr_model = LogisticRegression(random_state=42, max_iter=1000)
    lr_model.fit(X_train_scaled, y_train)
    y_pred_lr = lr_model.predict(X_test_scaled)
    y_prob_lr = lr_model.predict_proba(X_test_scaled)[:, 1]
    
    # Train Random Forest
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    rf_model.fit(X_train, y_train)
    y_pred_rf = rf_model.predict(X_test)
    y_prob_rf = rf_model.predict_proba(X_test)[:, 1]
    
    return {
        'lr_model': lr_model,
        'rf_model': rf_model,
        'scaler': scaler,
        'label_encoders': label_encoders,
        'X_test': X_test,
        'X_test_scaled': X_test_scaled,
        'y_test': y_test,
        'y_pred_lr': y_pred_lr,
        'y_prob_lr': y_prob_lr,
        'y_pred_rf': y_pred_rf,
        'y_prob_rf': y_prob_rf,
        'feature_names': X.columns.tolist(),
        'original_X': X,
        'original_y': y
    }


def display_metrics(y_true, y_pred, y_prob):
    """Calculate and display model metrics"""
    metrics = {
        'Accuracy': accuracy_score(y_true, y_pred),
        'Precision': precision_score(y_true, y_pred),
        'Recall': recall_score(y_true, y_pred),
        'F1-Score': f1_score(y_true, y_pred),
        'ROC-AUC': roc_auc_score(y_true, y_prob)
    }
    return metrics


def main():
    # Sidebar
    with st.sidebar:
        st.markdown("## 🎯 Navigation")
        page = st.radio(
            "",
            ["📊 Overview", "🔍 Data Explorer", "🤖 Model Performance", 
             "💰 Cost Analysis", "🎯 Predict Churn"],
            label_visibility="collapsed"
        )
        
        st.markdown("---")
        st.markdown("### 📈 Quick Stats")
        
    # Load data
    try:
        df_raw = load_data()
        df = preprocess_data(df_raw)
    except Exception as e:
        st.error(f"Error loading data: {e}")
        st.stop()
    
    # Display quick stats in sidebar
    with st.sidebar:
        st.metric("Total Customers", f"{len(df):,}")
        churn_rate = (df['Churn'] == 'Yes').mean() * 100
        st.metric("Churn Rate", f"{churn_rate:.1f}%")
        st.metric("Features", f"{df.shape[1] - 1}")
    
    # Page routing
    if page == "📊 Overview":
        show_overview(df)
    elif page == "🔍 Data Explorer":
        show_data_explorer(df)
    elif page == "🤖 Model Performance":
        show_model_performance(df)
    elif page == "💰 Cost Analysis":
        show_cost_analysis(df)
    elif page == "🎯 Predict Churn":
        show_prediction(df)


def show_overview(df):
    """Display the overview page"""
    st.markdown("# 📊 Customer Churn Risk Analytics")
    st.markdown("### Real-time insights for proactive customer retention")
    
    st.markdown("---")
    
    # Key metrics row
    col1, col2, col3, col4 = st.columns(4)
    
    churn_rate = (df['Churn'] == 'Yes').mean() * 100
    avg_tenure = df['tenure'].mean()
    avg_monthly = df['MonthlyCharges'].mean()
    total_revenue = df['TotalCharges'].sum()
    
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{len(df):,}</div>
            <div class="metric-label">Total Customers</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{churn_rate:.1f}%</div>
            <div class="metric-label">Churn Rate</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{avg_tenure:.0f}</div>
            <div class="metric-label">Avg Tenure (months)</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">${avg_monthly:.0f}</div>
            <div class="metric-label">Avg Monthly Charges</div>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Charts row
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown('<p class="section-header">📈 Churn Distribution</p>', unsafe_allow_html=True)
        fig, ax = plt.subplots(figsize=(8, 5))
        colors = ['#1dd1a1', '#ff6b6b']
        churn_counts = df['Churn'].value_counts()
        wedges, texts, autotexts = ax.pie(
            churn_counts.values, 
            labels=['Retained', 'Churned'],
            colors=colors,
            autopct='%1.1f%%',
            startangle=90,
            explode=(0, 0.05),
            shadow=True
        )
        for text in texts + autotexts:
            text.set_color('white')
            text.set_fontweight('bold')
        ax.set_facecolor('#0f0f23')
        fig.patch.set_facecolor('#0f0f23')
        st.pyplot(fig)
        plt.close()
    
    with col2:
        st.markdown('<p class="section-header">📊 Churn by Contract Type</p>', unsafe_allow_html=True)
        churn_by_contract = df.groupby('Contract')['Churn'].apply(lambda x: (x == 'Yes').mean() * 100)
        fig, ax = plt.subplots(figsize=(8, 5))
        colors = ['#ff6b6b', '#feca57', '#1dd1a1']
        bars = ax.bar(churn_by_contract.index, churn_by_contract.values, color=colors, edgecolor='white', linewidth=1.5)
        ax.set_ylabel('Churn Rate (%)', color='white', fontsize=12)
        ax.set_xlabel('Contract Type', color='white', fontsize=12)
        ax.tick_params(colors='white')
        ax.set_facecolor('#1a1a2e')
        fig.patch.set_facecolor('#0f0f23')
        ax.spines['bottom'].set_color('white')
        ax.spines['left'].set_color('white')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        for bar, val in zip(bars, churn_by_contract.values):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, 
                   f'{val:.1f}%', ha='center', color='white', fontweight='bold')
        st.pyplot(fig)
        plt.close()
    
    # Key insights
    st.markdown('<p class="section-header">💡 Key Insights</p>', unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class="insight-card">
            <h4>🔴 High Risk: Month-to-Month</h4>
            <p>Customers on month-to-month contracts have ~43% churn rate. 
            Consider incentivizing longer contracts.</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="insight-card">
            <h4>⚡ Fiber Optic Alert</h4>
            <p>Fiber optic customers show higher churn. Investigate service 
            quality and competitive pricing.</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="insight-card">
            <h4>🆕 New Customer Focus</h4>
            <p>Customers in first 12 months are most likely to leave. 
            Strengthen onboarding experience.</p>
        </div>
        """, unsafe_allow_html=True)


def show_data_explorer(df):
    """Display the data explorer page"""
    st.markdown("# 🔍 Data Explorer")
    st.markdown("### Explore customer data patterns and distributions")
    
    st.markdown("---")
    
    # Filters
    col1, col2, col3 = st.columns(3)
    with col1:
        contract_filter = st.multiselect("Contract Type", df['Contract'].unique(), default=df['Contract'].unique())
    with col2:
        internet_filter = st.multiselect("Internet Service", df['InternetService'].unique(), default=df['InternetService'].unique())
    with col3:
        churn_filter = st.multiselect("Churn Status", df['Churn'].unique(), default=df['Churn'].unique())
    
    # Apply filters
    df_filtered = df[
        (df['Contract'].isin(contract_filter)) &
        (df['InternetService'].isin(internet_filter)) &
        (df['Churn'].isin(churn_filter))
    ]
    
    st.markdown(f"**Showing {len(df_filtered):,} customers**")
    
    # Distribution charts
    st.markdown('<p class="section-header">📊 Feature Distributions</p>', unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.hist(df_filtered['tenure'], bins=30, color='#667eea', edgecolor='white', alpha=0.8)
        ax.set_title('Tenure Distribution', color='white', fontsize=14, fontweight='bold')
        ax.set_xlabel('Tenure (months)', color='white')
        ax.set_ylabel('Count', color='white')
        ax.tick_params(colors='white')
        ax.set_facecolor('#1a1a2e')
        fig.patch.set_facecolor('#0f0f23')
        ax.spines['bottom'].set_color('white')
        ax.spines['left'].set_color('white')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        st.pyplot(fig)
        plt.close()
    
    with col2:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.hist(df_filtered['MonthlyCharges'], bins=30, color='#764ba2', edgecolor='white', alpha=0.8)
        ax.set_title('Monthly Charges', color='white', fontsize=14, fontweight='bold')
        ax.set_xlabel('Monthly Charges ($)', color='white')
        ax.set_ylabel('Count', color='white')
        ax.tick_params(colors='white')
        ax.set_facecolor('#1a1a2e')
        fig.patch.set_facecolor('#0f0f23')
        ax.spines['bottom'].set_color('white')
        ax.spines['left'].set_color('white')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        st.pyplot(fig)
        plt.close()
    
    with col3:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.hist(df_filtered['TotalCharges'], bins=30, color='#1dd1a1', edgecolor='white', alpha=0.8)
        ax.set_title('Total Charges', color='white', fontsize=14, fontweight='bold')
        ax.set_xlabel('Total Charges ($)', color='white')
        ax.set_ylabel('Count', color='white')
        ax.tick_params(colors='white')
        ax.set_facecolor('#1a1a2e')
        fig.patch.set_facecolor('#0f0f23')
        ax.spines['bottom'].set_color('white')
        ax.spines['left'].set_color('white')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        st.pyplot(fig)
        plt.close()
    
    # Data table
    st.markdown('<p class="section-header">📋 Customer Data</p>', unsafe_allow_html=True)
    st.dataframe(df_filtered, use_container_width=True, height=400)


def show_model_performance(df):
    """Display model performance page"""
    st.markdown("# 🤖 Model Performance")
    st.markdown("### Comparing Logistic Regression vs Random Forest")
    
    st.markdown("---")
    
    with st.spinner("Training models... This may take a moment."):
        models = train_models(df)
    
    # Get metrics
    lr_metrics = display_metrics(models['y_test'], models['y_pred_lr'], models['y_prob_lr'])
    rf_metrics = display_metrics(models['y_test'], models['y_pred_rf'], models['y_prob_rf'])
    
    # Model comparison
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 📈 Logistic Regression")
        for metric, value in lr_metrics.items():
            st.metric(metric, f"{value:.4f}")
    
    with col2:
        st.markdown("### 🌲 Random Forest")
        for metric, value in rf_metrics.items():
            delta = value - lr_metrics[metric]
            st.metric(metric, f"{value:.4f}", delta=f"{delta:+.4f}")
    
    # ROC Curves
    st.markdown('<p class="section-header">📊 ROC Curves</p>', unsafe_allow_html=True)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Logistic Regression ROC
    fpr_lr, tpr_lr, _ = roc_curve(models['y_test'], models['y_prob_lr'])
    ax.plot(fpr_lr, tpr_lr, color='#667eea', linewidth=3, 
            label=f'Logistic Regression (AUC = {lr_metrics["ROC-AUC"]:.4f})')
    
    # Random Forest ROC
    fpr_rf, tpr_rf, _ = roc_curve(models['y_test'], models['y_prob_rf'])
    ax.plot(fpr_rf, tpr_rf, color='#1dd1a1', linewidth=3,
            label=f'Random Forest (AUC = {rf_metrics["ROC-AUC"]:.4f})')
    
    ax.plot([0, 1], [0, 1], 'k--', alpha=0.5, linewidth=2)
    ax.set_xlabel('False Positive Rate', color='white', fontsize=12)
    ax.set_ylabel('True Positive Rate', color='white', fontsize=12)
    ax.set_title('ROC Curves Comparison', color='white', fontsize=16, fontweight='bold')
    ax.legend(loc='lower right', facecolor='#1a1a2e', edgecolor='white', labelcolor='white')
    ax.tick_params(colors='white')
    ax.set_facecolor('#1a1a2e')
    fig.patch.set_facecolor('#0f0f23')
    ax.spines['bottom'].set_color('white')
    ax.spines['left'].set_color('white')
    ax.spines['top'].set_color('white')
    ax.spines['right'].set_color('white')
    ax.grid(True, alpha=0.2, color='white')
    st.pyplot(fig)
    plt.close()
    
    # Feature Importance
    st.markdown('<p class="section-header">🎯 Feature Importance (Random Forest)</p>', unsafe_allow_html=True)
    
    feature_importance = pd.DataFrame({
        'Feature': models['feature_names'],
        'Importance': models['rf_model'].feature_importances_
    }).sort_values('Importance', ascending=True).tail(10)
    
    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.barh(feature_importance['Feature'], feature_importance['Importance'], 
                   color='#764ba2', edgecolor='white')
    ax.set_xlabel('Importance', color='white', fontsize=12)
    ax.set_title('Top 10 Most Important Features', color='white', fontsize=14, fontweight='bold')
    ax.tick_params(colors='white')
    ax.set_facecolor('#1a1a2e')
    fig.patch.set_facecolor('#0f0f23')
    ax.spines['bottom'].set_color('white')
    ax.spines['left'].set_color('white')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    st.pyplot(fig)
    plt.close()


def show_cost_analysis(df):
    """Display cost analysis page"""
    st.markdown("# 💰 Business Cost Analysis")
    st.markdown("### Optimize decision threshold to minimize business costs")
    
    st.markdown("---")
    
    with st.spinner("Training models..."):
        models = train_models(df)
    
    # Cost inputs
    st.markdown("### 💵 Define Business Costs")
    col1, col2 = st.columns(2)
    
    with col1:
        cost_fn = st.slider("Cost of Missing a Churner (False Negative)", 100, 1000, 500, 50)
        st.markdown("*Lost revenue from customer leaving*")
    
    with col2:
        cost_fp = st.slider("Cost of Unnecessary Offer (False Positive)", 10, 200, 50, 10)
        st.markdown("*Cost of discount given to non-churner*")
    
    # Calculate costs for different thresholds
    thresholds = np.arange(0.2, 0.8, 0.05)
    results = []
    
    for thresh in thresholds:
        y_pred_thresh = (models['y_prob_rf'] >= thresh).astype(int)
        tn, fp, fn, tp = confusion_matrix(models['y_test'], y_pred_thresh).ravel()
        total_cost = (fn * cost_fn) + (fp * cost_fp)
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        
        results.append({
            'Threshold': thresh,
            'TP': tp, 'FP': fp, 'TN': tn, 'FN': fn,
            'Precision': precision,
            'Recall': recall,
            'Total Cost': total_cost
        })
    
    results_df = pd.DataFrame(results)
    optimal_idx = results_df['Total Cost'].idxmin()
    optimal_threshold = results_df.loc[optimal_idx, 'Threshold']
    
    # Display optimal threshold
    st.markdown(f"""
    <div class="metric-card" style="margin: 20px 0;">
        <div class="metric-value">{optimal_threshold:.2f}</div>
        <div class="metric-label">Optimal Decision Threshold</div>
        <p style="color: #a0a0b0; margin-top: 10px;">
            Minimizes total cost to ${results_df.loc[optimal_idx, 'Total Cost']:,.0f}
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Cost by threshold chart
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown('<p class="section-header">📊 Total Cost by Threshold</p>', unsafe_allow_html=True)
        fig, ax = plt.subplots(figsize=(8, 5))
        colors = ['#1dd1a1' if t == optimal_threshold else '#667eea' for t in results_df['Threshold']]
        ax.bar(results_df['Threshold'], results_df['Total Cost'], color=colors, width=0.04)
        ax.axvline(x=optimal_threshold, color='#ff6b6b', linestyle='--', linewidth=2, label=f'Optimal: {optimal_threshold:.2f}')
        ax.set_xlabel('Threshold', color='white', fontsize=12)
        ax.set_ylabel('Total Cost ($)', color='white', fontsize=12)
        ax.tick_params(colors='white')
        ax.set_facecolor('#1a1a2e')
        fig.patch.set_facecolor('#0f0f23')
        ax.spines['bottom'].set_color('white')
        ax.spines['left'].set_color('white')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.legend(facecolor='#1a1a2e', edgecolor='white', labelcolor='white')
        st.pyplot(fig)
        plt.close()
    
    with col2:
        st.markdown('<p class="section-header">📈 Precision-Recall Trade-off</p>', unsafe_allow_html=True)
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(results_df['Threshold'], results_df['Precision'], 'o-', color='#667eea', linewidth=2, markersize=6, label='Precision')
        ax.plot(results_df['Threshold'], results_df['Recall'], 's-', color='#1dd1a1', linewidth=2, markersize=6, label='Recall')
        ax.axvline(x=optimal_threshold, color='#ff6b6b', linestyle='--', linewidth=2, label=f'Optimal: {optimal_threshold:.2f}')
        ax.set_xlabel('Threshold', color='white', fontsize=12)
        ax.set_ylabel('Score', color='white', fontsize=12)
        ax.tick_params(colors='white')
        ax.set_facecolor('#1a1a2e')
        fig.patch.set_facecolor('#0f0f23')
        ax.spines['bottom'].set_color('white')
        ax.spines['left'].set_color('white')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.legend(facecolor='#1a1a2e', edgecolor='white', labelcolor='white')
        ax.grid(True, alpha=0.2, color='white')
        st.pyplot(fig)
        plt.close()
    
    # Display results table
    st.markdown('<p class="section-header">📋 Detailed Results</p>', unsafe_allow_html=True)
    display_df = results_df.round(3)
    display_df['Total Cost'] = display_df['Total Cost'].apply(lambda x: f"${x:,.0f}")
    st.dataframe(display_df, use_container_width=True)


def show_prediction(df):
    """Display the prediction page"""
    st.markdown("# 🎯 Churn Prediction")
    st.markdown("### Enter customer details to predict churn probability")
    
    st.markdown("---")
    
    with st.spinner("Loading models..."):
        models = train_models(df)
    
    # Create input form
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("#### 👤 Customer Info")
        gender = st.selectbox("Gender", ["Male", "Female"])
        senior_citizen = st.selectbox("Senior Citizen", ["No", "Yes"])
        partner = st.selectbox("Partner", ["No", "Yes"])
        dependents = st.selectbox("Dependents", ["No", "Yes"])
        tenure = st.slider("Tenure (months)", 0, 72, 12)
    
    with col2:
        st.markdown("#### 📞 Services")
        phone_service = st.selectbox("Phone Service", ["No", "Yes"])
        multiple_lines = st.selectbox("Multiple Lines", ["No", "Yes", "No phone service"])
        internet_service = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
        online_security = st.selectbox("Online Security", ["No", "Yes", "No internet service"])
        online_backup = st.selectbox("Online Backup", ["No", "Yes", "No internet service"])
        device_protection = st.selectbox("Device Protection", ["No", "Yes", "No internet service"])
        tech_support = st.selectbox("Tech Support", ["No", "Yes", "No internet service"])
    
    with col3:
        st.markdown("#### 💳 Billing")
        streaming_tv = st.selectbox("Streaming TV", ["No", "Yes", "No internet service"])
        streaming_movies = st.selectbox("Streaming Movies", ["No", "Yes", "No internet service"])
        contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
        paperless_billing = st.selectbox("Paperless Billing", ["No", "Yes"])
        payment_method = st.selectbox("Payment Method", 
            ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"])
        monthly_charges = st.number_input("Monthly Charges ($)", 0.0, 200.0, 70.0)
        total_charges = st.number_input("Total Charges ($)", 0.0, 10000.0, monthly_charges * tenure)
    
    st.markdown("---")
    
    if st.button("🔮 Predict Churn Risk", use_container_width=True):
        # Prepare input data
        input_data = {
            'gender': gender,
            'SeniorCitizen': 1 if senior_citizen == "Yes" else 0,
            'Partner': partner,
            'Dependents': dependents,
            'tenure': tenure,
            'PhoneService': phone_service,
            'MultipleLines': multiple_lines,
            'InternetService': internet_service,
            'OnlineSecurity': online_security,
            'OnlineBackup': online_backup,
            'DeviceProtection': device_protection,
            'TechSupport': tech_support,
            'StreamingTV': streaming_tv,
            'StreamingMovies': streaming_movies,
            'Contract': contract,
            'PaperlessBilling': paperless_billing,
            'PaymentMethod': payment_method,
            'MonthlyCharges': monthly_charges,
            'TotalCharges': total_charges
        }
        
        # Encode input
        input_encoded = input_data.copy()
        for col, le in models['label_encoders'].items():
            if col in input_encoded and col != 'SeniorCitizen':
                try:
                    input_encoded[col] = le.transform([input_encoded[col]])[0]
                except ValueError:
                    input_encoded[col] = 0
        
        # Create dataframe
        input_df = pd.DataFrame([input_encoded])
        
        # Ensure correct column order
        input_df = input_df[models['feature_names']]
        
        # Make prediction
        prob = models['rf_model'].predict_proba(input_df)[0][1]
        
        # Display result
        st.markdown("---")
        
        if prob >= 0.5:
            st.markdown(f"""
            <div class="prediction-result churn-high">
                <h2>⚠️ HIGH CHURN RISK</h2>
                <p style="font-size: 2rem; font-weight: bold;">{prob*100:.1f}%</p>
                <p>This customer has a high probability of churning.</p>
                <p><strong>Recommendation:</strong> Immediate retention intervention needed!</p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="prediction-result churn-low">
                <h2>✅ LOW CHURN RISK</h2>
                <p style="font-size: 2rem; font-weight: bold;">{prob*100:.1f}%</p>
                <p>This customer is likely to stay.</p>
                <p><strong>Recommendation:</strong> Continue monitoring, focus on satisfaction.</p>
            </div>
            """, unsafe_allow_html=True)
        
        # Risk factors
        st.markdown('<p class="section-header">📊 Risk Analysis</p>', unsafe_allow_html=True)
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            tenure_risk = "🔴 High" if tenure < 12 else ("🟡 Medium" if tenure < 24 else "🟢 Low")
            st.markdown(f"""
            <div class="insight-card">
                <h4>Tenure Risk: {tenure_risk}</h4>
                <p>{tenure} months with the company</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            contract_risk = "🔴 High" if contract == "Month-to-month" else ("🟡 Medium" if contract == "One year" else "🟢 Low")
            st.markdown(f"""
            <div class="insight-card">
                <h4>Contract Risk: {contract_risk}</h4>
                <p>{contract} contract type</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            charges_risk = "🔴 High" if monthly_charges > 80 else ("🟡 Medium" if monthly_charges > 50 else "🟢 Low")
            st.markdown(f"""
            <div class="insight-card">
                <h4>Price Sensitivity: {charges_risk}</h4>
                <p>${monthly_charges:.2f}/month</p>
            </div>
            """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
