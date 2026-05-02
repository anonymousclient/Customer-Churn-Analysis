# ============================================================
# Customer Churn Analysis System
# Main Streamlit Application
# ============================================================
# This app allows users to upload customer data (CSV/Excel),
# automatically cleans and transforms it, then displays an
# interactive dashboard with KPIs, charts, insights, and
# recommendations for reducing customer churn.
# ============================================================

# ----- IMPORT LIBRARIES -----
import streamlit as st          # Web app framework
import pandas as pd             # Data manipulation
import numpy as np              # Numerical operations
import plotly.express as px     # Interactive charts
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import warnings
warnings.filterwarnings("ignore")


# ============================================================
# PAGE CONFIGURATION
# ============================================================
st.set_page_config(
    page_title="Customer Churn Analysis",
    page_icon="📊",
    layout="wide",                # Use full width of screen
    initial_sidebar_state="expanded"
)


# ============================================================
# CUSTOM CSS STYLING
# ============================================================
def load_custom_css():
    """Inject custom CSS for a polished, professional look."""
    st.markdown("""
    <style>
        /* ---- Main container ---- */
        .main .block-container {
            padding-top: 1.5rem;
            padding-bottom: 2rem;
            background-color: #0B1120;
        }

        /* ---- Header banner ---- */
        .hero-banner {
            background: linear-gradient(135deg, #4F46E5 0%, #6366F1 50%, #818CF8 100%);
            padding: 2rem 2.5rem;
            border-radius: 16px;
            margin-bottom: 1.5rem;
            text-align: center;
            box-shadow: 0 8px 32px rgba(99, 102, 241, 0.2);
        }
        .hero-banner h1 {
            color: #ffffff;
            font-size: 2.4rem;
            margin: 0;
            font-weight: 800;
            letter-spacing: -0.5px;
        }
        .hero-banner p {
            color: rgba(255,255,255,0.9);
            font-size: 1.05rem;
            margin-top: 0.4rem;
        }

        /* ---- Buttons ---- */
        .stButton > button {
            background-color: #6366F1 !important;
            color: #ffffff !important;
            border-radius: 8px !important;
            border: none !important;
            font-weight: 600 !important;
            padding: 0.5rem 1rem !important;
            transition: all 0.3s ease !important;
        }
        .stButton > button:hover {
            background-color: #4F46E5 !important;
            box-shadow: 0 4px 12px rgba(99, 102, 241, 0.4) !important;
            transform: translateY(-2px);
        }

        /* ---- KPI metric cards ---- */
        .kpi-card {
            background: #111827;
            border: 1px solid rgba(255,255,255,0.05);
            border-radius: 14px;
            padding: 1.5rem;
            text-align: center;
            box-shadow: 0 4px 15px rgba(0,0,0,0.2);
            transition: transform 0.2s ease;
        }
        .kpi-card:hover {
            transform: translateY(-4px);
            border-color: rgba(99, 102, 241, 0.3);
        }
        .kpi-icon { font-size: 2rem; margin-bottom: 0.3rem; }
        .kpi-value {
            font-size: 2.2rem;
            font-weight: 800;
            color: #E5E7EB;
        }
        .kpi-label {
            font-size: 0.9rem;
            color: #9CA3AF;
            margin-top: 0.2rem;
            text-transform: uppercase;
            letter-spacing: 1px;
            font-weight: 600;
        }

        /* ---- Section headers ---- */
        .section-header {
            font-size: 1.5rem;
            font-weight: 700;
            color: #F9FAFB;
            margin-top: 2rem;
            margin-bottom: 1rem;
            border-left: 4px solid #6366F1;
            padding-left: 12px;
        }

        /* ---- Insight & Rec cards ---- */
        .insight-card, .rec-card {
            background: #111827;
            border-left: 4px solid #6366F1;
            border-radius: 8px;
            padding: 1rem 1.2rem;
            margin-bottom: 0.7rem;
            font-size: 0.95rem;
            color: #D1D5DB;
            box-shadow: 0 2px 8px rgba(0,0,0,0.15);
        }
        .rec-card { border-left: 4px solid #38BDF8; }

        /* ---- Sidebar styling ---- */
        section[data-testid="stSidebar"] {
            background-color: #111827 !important;
            border-right: 1px solid rgba(255,255,255,0.05);
        }
        section[data-testid="stSidebar"] .block-container {
            padding-top: 1rem;
        }

        /* ---- Divider ---- */
        .custom-divider {
            border: none;
            height: 1px;
            background: linear-gradient(90deg, transparent, rgba(99, 102, 241, 0.3), transparent);
            margin: 1.5rem 0;
        }

        /* Hide default Streamlit footer */
        footer { visibility: hidden; }
    </style>
    """, unsafe_allow_html=True)


# ============================================================
# DATA CLEANING FUNCTION
# ============================================================
def clean_data(df):
    """
    Automatically clean the uploaded dataset and ensure all required
    columns for analysis are present. If a column is missing, it
    generates synthetic data for demonstration purposes.
    """
    cleaned = df.copy()
    generated_cols = []

    # --- Step 0: Standardize existing column names (lowercase + underscore) ---
    cleaned.columns = [c.lower().replace(' ', '_') for c in cleaned.columns]

    # --- Step 1: Define Required Columns & Generation Logic ---
    # We define what each column needs to look like if it's missing
    required_logic = {
        'churn': lambda size: np.random.choice([0, 1], size=size, p=[0.8, 0.2]),
        'credit_score': lambda size: np.random.randint(350, 850, size=size),
        'age': lambda size: np.random.randint(18, 85, size=size),
        'balance': lambda size: np.random.uniform(0, 220000, size=size).round(2),
        'tenure': lambda size: np.random.randint(0, 11, size=size),
        'products_number': lambda size: np.random.randint(1, 5, size=size),
        'credit_card': lambda size: np.random.choice([0, 1], size=size),
        'active_member': lambda size: np.random.choice([0, 1], size=size, p=[0.4, 0.6]),
        'gender': lambda size: np.random.choice(['Male', 'Female'], size=size),
        'country': lambda size: np.random.choice(['France', 'Spain', 'Germany'], size=size)
    }

    # Check and generate missing columns
    for col, logic in required_logic.items():
        if col not in cleaned.columns:
            cleaned[col] = logic(len(cleaned))
            generated_cols.append(col.replace('_', ' ').title())

    # --- Step 2: Handle missing values in existing columns ---
    for col in cleaned.columns:
        if cleaned[col].isnull().sum() > 0:
            if cleaned[col].dtype in ['int64', 'float64']:
                cleaned[col].fillna(cleaned[col].mean(), inplace=True)
            else:
                cleaned[col].fillna(cleaned[col].mode()[0], inplace=True)

    # --- Step 3: Final Renaming for UI Readability ---
    rename_map = {
        'customer_id': 'Customer ID',
        'credit_score': 'Credit Score',
        'country': 'Country',
        'gender': 'Gender',
        'age': 'Age',
        'tenure': 'Tenure',
        'balance': 'Balance',
        'products_number': 'Products',
        'credit_card': 'Credit Card',
        'active_member': 'Active Member',
        'churn': 'Churn Status'
    }
    cleaned.rename(columns={k: v for k, v in rename_map.items()
                            if k in cleaned.columns}, inplace=True)

    # --- Step 4: Convert binary columns to readable labels ---
    if 'Churn Status' in cleaned.columns:
        cleaned['Churn Status'] = cleaned['Churn Status'].map(
            {1: 'Yes', 0: 'No', '1': 'Yes', '0': 'No'}
        ).fillna(cleaned['Churn Status'])

    if 'Active Member' in cleaned.columns:
        cleaned['Active Member'] = cleaned['Active Member'].map(
            {1: 'Yes', 0: 'No', '1': 'Yes', '0': 'No'}
        ).fillna(cleaned['Active Member'])

    if 'Credit Card' in cleaned.columns:
        cleaned['Credit Card'] = cleaned['Credit Card'].map(
            {1: 'Yes', 0: 'No', '1': 'Yes', '0': 'No'}
        ).fillna(cleaned['Credit Card'])

    return cleaned, generated_cols


# ============================================================
# DATA TRANSFORMATION FUNCTION
# ============================================================
def transform_data(df):
    """
    Create new grouped columns for deeper analysis.
    - Age Group:          Young / Middle-aged / Senior
    - Balance Group:      Low / Medium / High
    - Credit Score Group: Low / Medium / High
    """
    transformed = df.copy()

    # --- Age Group ---
    if 'Age' in transformed.columns:
        transformed['Age Group'] = pd.cut(
            transformed['Age'],
            bins=[0, 30, 50, 120],
            labels=['Young (18-30)', 'Middle-aged (31-50)', 'Senior (51+)']
        )

    # --- Balance Group ---
    if 'Balance' in transformed.columns:
        transformed['Balance Group'] = pd.cut(
            transformed['Balance'],
            bins=[-1, 50000, 150000, float('inf')],
            labels=['Low (<50k)', 'Medium (50k-150k)', 'High (>150k)']
        )

    # --- Credit Score Group ---
    if 'Credit Score' in transformed.columns:
        transformed['Credit Score Group'] = pd.cut(
            transformed['Credit Score'],
            bins=[0, 500, 700, 900],
            labels=['Low (<500)', 'Medium (500-700)', 'High (>700)']
        )

    return transformed


# ============================================================
# CHART CREATION HELPERS
# ============================================================
CHART_COLORS = ['#6366F1', '#38BDF8', '#818CF8', '#A78BFA',
                '#2DD4BF', '#F472B6', '#C084FC', '#60A5FA']


def create_bar_chart(data, x_col, title):
    """Create a styled bar chart showing churn counts by category."""
    chart_data = data.groupby([x_col, 'Churn Status']).size().reset_index(name='Count')
    fig = px.bar(
        chart_data, x=x_col, y='Count', color='Churn Status',
        barmode='group', title=title,
        color_discrete_map={'Yes': '#6366F1', 'No': '#38BDF8'},
        text='Count'
    )
    fig.update_layout(
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font_color='#E5E7EB',
        title_font_size=16,
        title_font_color='#E5E7EB',
        legend=dict(bgcolor='rgba(0,0,0,0)'),
        margin=dict(t=50, b=40, l=40, r=20),
        xaxis=dict(showgrid=False),
        yaxis=dict(showgrid=True, gridcolor='rgba(255,255,255,0.05)')
    )
    fig.update_traces(textposition='outside', textfont_size=11)
    return fig


def create_pie_chart(data, col, title):
    """Create a styled pie/donut chart for categorical distribution."""
    chart_data = data.groupby([col, 'Churn Status']).size().reset_index(name='Count')
    churned = chart_data[chart_data['Churn Status'] == 'Yes']
    if churned.empty:
        churned = chart_data

    fig = px.pie(
        churned, names=col, values='Count', title=title,
        color_discrete_sequence=CHART_COLORS, hole=0.45
    )
    fig.update_layout(
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font_color='#E5E7EB',
        title_font_size=16,
        title_font_color='#E5E7EB',
        legend=dict(bgcolor='rgba(0,0,0,0)'),
        margin=dict(t=50, b=20, l=20, r=20)
    )
    fig.update_traces(textinfo='percent+label', textfont_size=12)
    return fig


# ============================================================
# INSIGHT GENERATION
# ============================================================
def generate_insights(data):
    """Analyse the data and return a list of text insights."""
    insights = []

    if 'Age Group' in data.columns and 'Churn Status' in data.columns:
        churn_by_age = data[data['Churn Status'] == 'Yes']['Age Group'].value_counts()
        if not churn_by_age.empty:
            top_age = churn_by_age.index[0]
            insights.append(
                f"📌 **{top_age}** customers have the highest churn count "
                f"({churn_by_age.iloc[0]} churned)."
            )

    if 'Gender' in data.columns and 'Churn Status' in data.columns:
        churn_by_gender = data[data['Churn Status'] == 'Yes']['Gender'].value_counts()
        if not churn_by_gender.empty:
            top_gender = churn_by_gender.index[0]
            insights.append(
                f"👤 **{top_gender}** customers show a higher tendency to churn "
                f"({churn_by_gender.iloc[0]} churned)."
            )

    if 'Balance Group' in data.columns and 'Churn Status' in data.columns:
        churn_by_bal = data[data['Churn Status'] == 'Yes']['Balance Group'].value_counts()
        if not churn_by_bal.empty:
            top_bal = churn_by_bal.index[0]
            insights.append(
                f"💰 Customers in the **{top_bal}** balance group churn the most "
                f"({churn_by_bal.iloc[0]} churned)."
            )

    if 'Credit Score Group' in data.columns and 'Churn Status' in data.columns:
        churn_by_cs = data[data['Churn Status'] == 'Yes']['Credit Score Group'].value_counts()
        if not churn_by_cs.empty:
            top_cs = churn_by_cs.index[0]
            insights.append(
                f"📊 Customers with **{top_cs}** credit scores churn the most "
                f"({churn_by_cs.iloc[0]} churned)."
            )

    if 'Active Member' in data.columns and 'Churn Status' in data.columns:
        inactive_churn = data[(data['Active Member'] == 'No') &
                             (data['Churn Status'] == 'Yes')].shape[0]
        active_churn = data[(data['Active Member'] == 'Yes') &
                            (data['Churn Status'] == 'Yes')].shape[0]
        if inactive_churn > active_churn:
            insights.append(
                f"⚠️ **Inactive members** churn significantly more "
                f"({inactive_churn}) than active members ({active_churn})."
            )

    if 'Country' in data.columns and 'Churn Status' in data.columns:
        churn_by_country = data[data['Churn Status'] == 'Yes']['Country'].value_counts()
        if not churn_by_country.empty:
            top_country = churn_by_country.index[0]
            insights.append(
                f"🌍 **{top_country}** has the highest number of churned customers "
                f"({churn_by_country.iloc[0]})."
            )

    if not insights:
        insights.append("ℹ️ Upload a valid dataset to generate insights.")

    return insights


# ============================================================
# RECOMMENDATIONS
# ============================================================
RECOMMENDATIONS = [
    ("🎁", "Offer Loyalty Incentives",
     "Provide special discounts, cashback, or rewards to high-risk "
     "customer segments to encourage retention."),
    ("📱", "Boost Engagement for Inactive Users",
     "Send personalised notifications, emails, or in-app messages "
     "to re-engage customers who have become inactive."),
    ("🎯", "Customised Plans for At-Risk Groups",
     "Design tailored product bundles or pricing plans for age groups "
     "and balance groups that exhibit high churn rates."),
    ("📞", "Proactive Customer Support",
     "Reach out to customers showing early signs of disengagement "
     "through dedicated relationship managers."),
    ("📊", "Monitor Credit Score Trends",
     "Track changes in credit scores and proactively offer financial "
     "advice or flexible products to customers with declining scores."),
    ("🌟", "Improve Onboarding Experience",
     "Ensure new customers receive a smooth onboarding journey with "
     "tutorials, welcome bonuses, and dedicated support."),
]


# ============================================================
# MACHINE LEARNING MODEL
# ============================================================
@st.cache_resource
def train_models(data):
    """
    Train Random Forest, KNN, and K-Means models.
    """
    # 1. DATA PREPARATION FOR ML
    ml_data = data.copy()
    
    # Map back target and binary variables
    ml_data['Churn Status'] = ml_data['Churn Status'].map({'Yes': 1, 'No': 0})
    ml_data['Active Member'] = ml_data['Active Member'].map({'Yes': 1, 'No': 0})
    ml_data['Credit Card'] = ml_data['Credit Card'].map({'Yes': 1, 'No': 0})
    ml_data['Gender'] = ml_data['Gender'].map({'Male': 1, 'Female': 0})
    
    # 2. FEATURE SELECTION (Country excluded)
    features = ['Credit Score', 'Age', 'Tenure', 'Balance', 'Products', 'Credit Card', 'Active Member', 'Gender']
    num_features = ['Credit Score', 'Age', 'Tenure', 'Balance', 'Products']
    
    # Ensure no missing values remain in features
    ml_data = ml_data.dropna(subset=features + ['Churn Status'])
    
    X = ml_data[features]
    y = ml_data['Churn Status']
    
    # 3. TRAIN-TEST SPLIT
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Scaler for KNN and K-Means
    scaler = StandardScaler()
    X_train_scaled = X_train.copy()
    X_test_scaled = X_test.copy()
    X_scaled_all = X.copy()
    
    X_train_scaled[num_features] = scaler.fit_transform(X_train[num_features])
    X_test_scaled[num_features] = scaler.transform(X_test[num_features])
    X_scaled_all[num_features] = scaler.transform(X[num_features])
    
    # 4. RANDOM FOREST (Unscaled)
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_model.fit(X_train, y_train)
    rf_pred = rf_model.predict(X_test)
    rf_metrics = {
        'accuracy': accuracy_score(y_test, rf_pred),
        'conf_matrix': confusion_matrix(y_test, rf_pred),
        'report': classification_report(y_test, rf_pred, output_dict=True)
    }
    
    # 5. KNN (Scaled)
    knn_model = KNeighborsClassifier(n_neighbors=5)
    knn_model.fit(X_train_scaled, y_train)
    knn_pred = knn_model.predict(X_test_scaled)
    knn_metrics = {
        'accuracy': accuracy_score(y_test, knn_pred),
        'conf_matrix': confusion_matrix(y_test, knn_pred),
        'report': classification_report(y_test, knn_pred, output_dict=True)
    }
    
    # 6. K-MEANS CLUSTERING (Scaled)
    kmeans_model = KMeans(n_clusters=3, random_state=42)
    kmeans_model.fit(X_scaled_all)
    cluster_labels = kmeans_model.labels_
    
    return rf_model, knn_model, kmeans_model, scaler, rf_metrics, knn_metrics, cluster_labels, features, num_features


# ============================================================
# MAIN APPLICATION
# ============================================================
def main():
    """Entry point – renders the full Streamlit dashboard."""
    load_custom_css()

    # ---- Hero Banner ----
    st.markdown("""
    <div class="hero-banner">
        <h1>📊 Customer Churn Analysis System</h1>
        <p>Upload your dataset • Analyse patterns • Reduce churn</p>
    </div>
    """, unsafe_allow_html=True)

    # ========================================================
    # SIDEBAR – File Upload & Filters
    # ========================================================
    with st.sidebar:
        st.markdown("## 📂 Data Upload")
        uploaded_file = st.file_uploader(
            "Upload CSV or Excel file",
            type=["csv", "xlsx", "xls"],
            help="Supported formats: .csv, .xlsx, .xls"
        )
        st.markdown("<hr class='custom-divider'>", unsafe_allow_html=True)

    # ---- Load data ----
    if uploaded_file is None:
        # Show a friendly landing state
        st.markdown("""
        <div style="text-align:center; padding:3rem 1rem;">
            <p style="font-size:4rem;">📁</p>
            <h3 style="color:#c8d6e5;">No dataset uploaded yet</h3>
            <p style="color:#8899aa;">
                Use the sidebar to upload a <b>CSV</b> or <b>Excel</b> file.<br>
                A sample dataset (<code>sample_dataset.csv</code>) is included
                in the project folder for quick testing.
            </p>
        </div>
        """, unsafe_allow_html=True)
        return  # Stop here until a file is uploaded

    # ---- Read file ----
    try:
        if uploaded_file.name.endswith('.csv'):
            raw_df = pd.read_csv(uploaded_file)
        else:
            raw_df = pd.read_excel(uploaded_file)
    except Exception as e:
        st.error(f"❌ Could not read the file: {e}")
        return

    # ---- Validate required columns (Legacy check removed for generalized support) ----
    # The app now auto-fills missing columns.

    # ========================================================
    # DASHBOARD OVERVIEW (Replaces Raw Preview)
    # ========================================================
    st.markdown("<p class='section-header'>📊 Dashboard Overview</p>",
                unsafe_allow_html=True)
    st.dataframe(raw_df.head(10), use_container_width=True)
    st.caption(f"Shape: {raw_df.shape[0]} rows × {raw_df.shape[1]} columns")

    st.markdown("<hr class='custom-divider'>", unsafe_allow_html=True)

    # ========================================================
    # CLEANING & TRANSFORMATION
    # ========================================================
    cleaned_df, gen_cols = clean_data(raw_df)
    transformed_df = transform_data(cleaned_df)

    # Show warning if columns were generated
    if gen_cols:
        with st.sidebar:
            st.warning(f"⚠️ **Missing columns detected!**\n\nWe auto-generated values for: "
                       f"*{', '.join(gen_cols)}* to make the dashboard work.")

    st.markdown("<hr class='custom-divider'>", unsafe_allow_html=True)

    # ========================================================
    # SIDEBAR FILTERS (built after cleaning so we know categories)
    # ========================================================
    with st.sidebar:
        st.markdown("## 🔍 Filters")

        # Gender filter
        if 'Gender' in transformed_df.columns:
            genders = ['All'] + sorted(transformed_df['Gender'].dropna().unique().tolist())
            sel_gender = st.selectbox("Gender", genders)
        else:
            sel_gender = 'All'

        # Country filter
        if 'Country' in transformed_df.columns:
            countries = ['All'] + sorted(transformed_df['Country'].dropna().unique().tolist())
            sel_country = st.selectbox("Country", countries)
        else:
            sel_country = 'All'

        # Activity filter
        if 'Active Member' in transformed_df.columns:
            activities = ['All'] + sorted(transformed_df['Active Member'].dropna().unique().tolist())
            sel_activity = st.selectbox("Activity Status", activities)
        else:
            sel_activity = 'All'

        st.markdown("<hr class='custom-divider'>", unsafe_allow_html=True)
        
        # Model Selection
        st.markdown("## 🤖 Model Selection")
        selected_model = st.selectbox(
            "Choose Analysis Model",
            ["Random Forest", "K-Nearest Neighbors (KNN)", "K-Means Clustering"]
        )

        st.markdown("<hr class='custom-divider'>", unsafe_allow_html=True)
        st.markdown(
            "<small style='color:#556677;'>Built with ❤️ using Streamlit</small>",
            unsafe_allow_html=True
        )

    # ---- Apply filters ----
    filtered_df = transformed_df.copy()
    if sel_gender != 'All' and 'Gender' in filtered_df.columns:
        filtered_df = filtered_df[filtered_df['Gender'] == sel_gender]
    if sel_country != 'All' and 'Country' in filtered_df.columns:
        filtered_df = filtered_df[filtered_df['Country'] == sel_country]
    if sel_activity != 'All' and 'Active Member' in filtered_df.columns:
        filtered_df = filtered_df[filtered_df['Active Member'] == sel_activity]

    # ========================================================
    # KPI SECTION
    # ========================================================

    total = len(filtered_df)
    if 'Churn Status' in filtered_df.columns:
        churned = filtered_df[filtered_df['Churn Status'] == 'Yes'].shape[0]
        retained = total - churned
        churn_rate = round((churned / total) * 100, 1) if total > 0 else 0
    else:
        churned, retained, churn_rate = 0, total, 0

    k1, k2, k3, k4 = st.columns(4)
    with k1:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-icon">👥</div>
            <div class="kpi-value">{total:,}</div>
            <div class="kpi-label">Total Customers</div>
        </div>""", unsafe_allow_html=True)
    with k2:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-icon">❌</div>
            <div class="kpi-value">{churned:,}</div>
            <div class="kpi-label">Churned</div>
        </div>""", unsafe_allow_html=True)
    with k3:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-icon">✅</div>
            <div class="kpi-value">{retained:,}</div>
            <div class="kpi-label">Retained</div>
        </div>""", unsafe_allow_html=True)
    with k4:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-icon">📉</div>
            <div class="kpi-value">{churn_rate}%</div>
            <div class="kpi-label">Churn Rate</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<hr class='custom-divider'>", unsafe_allow_html=True)

    # ========================================================
    # DASHBOARD CHARTS
    # ========================================================
    st.markdown("<p class='section-header'>📈 Churn Analysis</p>",
                unsafe_allow_html=True)

    if 'Churn Status' not in filtered_df.columns:
        st.info("ℹ️ No 'Churn Status' column found – charts cannot render.")
    else:
        # Row 1: Age Group + Gender
        c1, c2 = st.columns(2)
        with c1:
            if 'Age Group' in filtered_df.columns:
                st.plotly_chart(
                    create_bar_chart(filtered_df, 'Age Group',
                                    '🎂 Churn by Age Group'),
                    use_container_width=True
                )
        with c2:
            if 'Gender' in filtered_df.columns:
                st.plotly_chart(
                    create_pie_chart(filtered_df, 'Gender',
                                    '👤 Churn Distribution by Gender'),
                    use_container_width=True
                )

        # Row 2: Balance Group + Credit Score Group
        c3, c4 = st.columns(2)
        with c3:
            if 'Balance Group' in filtered_df.columns:
                st.plotly_chart(
                    create_bar_chart(filtered_df, 'Balance Group',
                                    '💰 Churn by Balance Group'),
                    use_container_width=True
                )
        with c4:
            if 'Credit Score Group' in filtered_df.columns:
                st.plotly_chart(
                    create_bar_chart(filtered_df, 'Credit Score Group',
                                    '📊 Churn by Credit Score Group'),
                    use_container_width=True
                )

        # Row 3: Activity Status + Country
        c5, c6 = st.columns(2)
        with c5:
            if 'Active Member' in filtered_df.columns:
                st.plotly_chart(
                    create_bar_chart(filtered_df, 'Active Member',
                                    '🏃 Churn by Activity Status'),
                    use_container_width=True
                )
        with c6:
            if 'Country' in filtered_df.columns:
                st.plotly_chart(
                    create_pie_chart(filtered_df, 'Country',
                                    '🌍 Churn Distribution by Country'),
                    use_container_width=True
                )

    st.markdown("<hr class='custom-divider'>", unsafe_allow_html=True)

    # ========================================================
    # INSIGHTS SECTION
    # ========================================================
    st.markdown("<p class='section-header'>📌 Insights & Recommendations</p>",
                unsafe_allow_html=True)

    insights = generate_insights(filtered_df)
    for insight in insights:
        st.markdown(
            f"<div class='insight-card'>{insight}</div>",
            unsafe_allow_html=True
        )

    st.markdown("<hr class='custom-divider'>", unsafe_allow_html=True)

    st.markdown("### 🚀 Recommendations", unsafe_allow_html=True)

    for icon, title, desc in RECOMMENDATIONS:
        st.markdown(
            f"<div class='rec-card'><b>{icon} {title}</b><br>{desc}</div>",
            unsafe_allow_html=True
        )

    # ---- Footer ----
    st.markdown("<hr class='custom-divider'>", unsafe_allow_html=True)
    
    # ========================================================
    # MACHINE LEARNING & PREDICTION
    # ========================================================
    st.markdown("<p class='section-header'>🔮 Prediction & Modeling</p>", unsafe_allow_html=True)
    
    # Train the ML models on the cleaned data
    models = train_models(cleaned_df)
    rf_model, knn_model, kmeans_model, scaler, rf_metrics, knn_metrics, cluster_labels, ml_features, num_features = models
    
    # Add cluster labels to dataframe for visualizations
    cleaned_df['Cluster'] = cluster_labels
    cleaned_df['Cluster'] = cleaned_df['Cluster'].map({0: 'Cluster A', 1: 'Cluster B', 2: 'Cluster C'})

    # Model Comparison Section
    st.markdown("### Model Comparison")
    
    comp_col1, comp_col2 = st.columns([1, 2])
    
    with comp_col1:
        st.dataframe(pd.DataFrame({
            'Model': ['Random Forest', 'KNN'],
            'Accuracy': [f"{rf_metrics['accuracy']*100:.2f}%", f"{knn_metrics['accuracy']*100:.2f}%"]
        }), hide_index=True)
        st.caption("ℹ️ K-Means is used for clustering and not directly comparable in accuracy.")
        
        # Simple Insight
        best_model = "Random Forest" if rf_metrics['accuracy'] > knn_metrics['accuracy'] else "KNN"
        st.info(f"💡 **Insight:** {best_model} performs better on this dataset.")
        
    with comp_col2:
        comp_df = pd.DataFrame({
            'Model': ['Random Forest', 'KNN'],
            'Accuracy': [rf_metrics['accuracy'], knn_metrics['accuracy']]
        })
        fig_comp = px.bar(comp_df, x='Model', y='Accuracy', title='Accuracy Comparison',
                         color='Model', color_discrete_sequence=['#6366F1', '#38BDF8'],
                         text=comp_df['Accuracy'].apply(lambda x: f"{x*100:.1f}%"))
        fig_comp.update_layout(yaxis_range=[0, 1], height=300, margin=dict(t=40, b=0, l=0, r=0), 
                              paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font_color='#E5E7EB')
        fig_comp.update_traces(textposition='outside')
        st.plotly_chart(fig_comp, use_container_width=True)

    st.markdown("---")

    if selected_model in ["Random Forest", "K-Nearest Neighbors (KNN)"]:
        st.markdown(f"### 🔮 Predict Customer Churn using {selected_model}")
        
        metrics = rf_metrics if selected_model == "Random Forest" else knn_metrics
        active_model = rf_model if selected_model == "Random Forest" else knn_model
        
        with st.expander(f"📊 View {selected_model} Performance Metrics"):
            st.write(f"**Accuracy Score:** {metrics['accuracy'] * 100:.2f}%")
            st.write("**Classification Report:**")
            st.dataframe(pd.DataFrame(metrics['report']).transpose())
            st.write("**Confusion Matrix:**")
            st.write(metrics['conf_matrix'])

        st.markdown("#### Enter Customer Details")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            p_age = st.number_input("Age", min_value=18, max_value=100, value=30)
            p_credit_score = st.number_input("Credit Score", min_value=300, max_value=850, value=600)
            p_gender = st.selectbox("Gender", ["Male", "Female"])
        with col2:
            p_balance = st.number_input("Balance", min_value=0.0, value=50000.0)
            p_tenure = st.number_input("Tenure (Years)", min_value=0, max_value=20, value=5)
            p_products = st.selectbox("Number of Products", [1, 2, 3, 4])
        with col3:
            p_credit_card = st.radio("Has Credit Card?", ["Yes", "No"])
            p_active = st.radio("Is Active Member?", ["Yes", "No"])
            
        if st.button("Predict Churn Risk", type="primary"):
            input_data = {
                'Credit Score': p_credit_score,
                'Age': p_age,
                'Tenure': p_tenure,
                'Balance': p_balance,
                'Products': p_products,
                'Credit Card': 1 if p_credit_card == "Yes" else 0,
                'Active Member': 1 if p_active == "Yes" else 0,
                'Gender': 1 if p_gender == "Male" else 0
            }
            input_df = pd.DataFrame([input_data])[ml_features]
            
            # Apply scaling if KNN
            if selected_model == "K-Nearest Neighbors (KNN)":
                input_df[num_features] = scaler.transform(input_df[num_features])
            
            prediction = active_model.predict(input_df)[0]
            prob = active_model.predict_proba(input_df)[0]
            churn_prob = prob[1] * 100
            
            st.markdown("### Prediction Result")
            if prediction == 1:
                st.error(f"🚨 **High Risk: This customer is likely to churn.** (Probability: {churn_prob:.1f}%)")
            else:
                st.success(f"✅ **Low Risk: This customer is likely to stay.** (Churn Probability: {churn_prob:.1f}%)")

    elif selected_model == "K-Means Clustering":
        st.markdown("### 🧩 Customer Segmentation using K-Means")
        st.markdown("We have grouped customers into 3 clusters based on their features. Let's analyse their behaviour.")
        
        # Scatter Plot
        fig_cluster = px.scatter(cleaned_df, x='Balance', y='Credit Score', color='Cluster', 
                                 title='Customer Clusters (Balance vs Credit Score)',
                                 color_discrete_sequence=['#6366F1', '#38BDF8', '#A78BFA'],
                                 opacity=0.7)
        fig_cluster.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font_color='#E5E7EB')
        st.plotly_chart(fig_cluster, use_container_width=True)
        
        # Cluster Characteristics
        st.markdown("#### Cluster Characteristics")
        numeric_cols = ['Credit Score', 'Age', 'Tenure', 'Balance', 'Products']
        cluster_summary = cleaned_df.groupby('Cluster')[numeric_cols].mean().round(2)
        
        # Calculate churn percentage per cluster
        churn_pct = cleaned_df.groupby('Cluster')['Churn Status'].apply(lambda x: (x == 'Yes').mean() * 100).round(1)
        cluster_summary['Churn Rate (%)'] = churn_pct
        
        st.dataframe(cluster_summary, use_container_width=True)
        
        # Identify highest risk cluster
        high_risk_cluster = cluster_summary['Churn Rate (%)'].idxmax()
        st.warning(f"⚠️ **{high_risk_cluster}** is the highest risk segment with a churn rate of {cluster_summary.loc[high_risk_cluster, 'Churn Rate (%)']}%. These customers require immediate attention.")

    st.markdown("<hr class='custom-divider'>", unsafe_allow_html=True)
    st.markdown(
        "<p style='text-align:center; color:#556677; font-size:0.85rem;'>"
        "© 2026 Customer Churn Analysis System • Built with Streamlit</p>",
        unsafe_allow_html=True
    )


# ============================================================
# RUN THE APP
# ============================================================
if __name__ == "__main__":
    main()
