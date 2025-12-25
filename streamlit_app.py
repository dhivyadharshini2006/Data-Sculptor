import streamlit as st
import pandas as pd
import os
import numpy as np

# Import your existing modular AI-driven functions
from utils.data_preprocessing import clean_data
from utils.data_insights import generate_insights
from utils.data_visualization import save_heatmap, save_trend_plot, save_distribution_plot
from utils.data_outliers import detect_outliers
from utils.data_quality import calculate_quality_score
from utils.data_ml import train_model

# --- Page Configuration ---
st.set_page_config(
    page_title="Data Sculptor",
    layout="wide",
    initial_sidebar_state="expanded"
)
# No authentication — start directly on the app pages. All special symbols removed.

# Ensure required folders exist for the visualization functions
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
os.makedirs(os.path.join(BASE_DIR, 'static/images'), exist_ok=True)
os.makedirs(os.path.join(BASE_DIR, 'uploads'), exist_ok=True)

# --- No authentication: ensure any leftover session keys are initialized ---
if "df" not in st.session_state:
    st.session_state.df = None
if "results" not in st.session_state:
    st.session_state.results = None
if "uploaded_filename" not in st.session_state:
    st.session_state.uploaded_filename = None

# --- Streamlit Cloud Caching Method ---
@st.cache_data(show_spinner=False)
def perform_ai_sculpting(df):
    """Orchestrates the modular utility functions with caching for performance."""
    df_cleaned, missing, missing_percent, duplicates, invalid_dates = clean_data(df)
    quality_score, quality_grade = calculate_quality_score(df_cleaned)
    total_outliers, outlier_counts = detect_outliers(df_cleaned)
    insights = generate_insights(df_cleaned)
    return {
        "df_cleaned": df_cleaned,
        "missing": missing,
        "duplicates": duplicates,
        "invalid_dates": invalid_dates,
        "quality_score": quality_score,
        "quality_grade": quality_grade,
        "total_outliers": total_outliers,
        "insights": insights
    }

# --- Custom CSS for "Shock" Factor ---
st.markdown("""
    <style>
    /* Main background and font */
    .main {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
    }
    /* Professional Card Styling */
    div[data-testid="stMetricValue"] {
        font-size: 2rem;
        color: #1E3A8A;
    }
    .metric-card {
        background: rgba(255, 255, 255, 0.7);
        padding: 20px;
        border-radius: 15px;
        border: 1px solid rgba(255, 255, 255, 0.3);
        box-shadow: 0 8px 32px 0 rgba(31, 38, 135, 0.1);
        backdrop-filter: blur(4px);
    }
    /* Sidebar styling */
    .css-1d391kg {
        background-color: #1E293B;
    }
    /* Success message styling */
    .stSuccess {
        border-left: 5px solid #10B981;
    }
    </style>
    """, unsafe_allow_html=True)

# --- Header ---
col1, col2 = st.columns([10, 2])
with col1:
    st.title("DATA SCULPTOR")
    st.markdown("**AI-Driven Data Cleaning & Analysis Tool**")
with col2:
    st.write("")
    st.write("")
st.markdown("---")

# --- Sidebar ---
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/2103/2103633.png", width=100)
    st.header("Control Panel ⚙️")
    st.markdown("<div style='margin-bottom:6px'><strong>Control</strong></div>", unsafe_allow_html=True)
    st.markdown("---")
    # uploader moved to main page center as requested
    st.markdown("---")
    st.info("Pro Tip: Upload datasets with missing values or outliers to see the AI's full potential.")

# --- Initialize session state for data ---
if "df" not in st.session_state:
    st.session_state.df = None
if "results" not in st.session_state:
    st.session_state.results = None

# --- Centered uploader (main page) ---
uploaded_file = None
if st.session_state.df is None:
    col_l, col_m, col_r = st.columns([1, 2, 1])
    with col_m:
        st.markdown("<div style='text-align:center; margin-top:30px'>", unsafe_allow_html=True)
        uploaded_file = st.file_uploader("Upload CSV file to begin", type=['csv'], key='center_uploader')
        st.markdown("</div>", unsafe_allow_html=True)
    if uploaded_file is not None:
        try:
            st.session_state.df = pd.read_csv(uploaded_file)
            st.session_state.uploaded_filename = uploaded_file.name
            # save a copy to uploads folder
            try:
                upload_path = os.path.join(BASE_DIR, 'uploads', uploaded_file.name)
                with open(upload_path, 'wb') as f:
                    f.write(uploaded_file.getbuffer())
            except Exception:
                pass
            st.success(f"Loaded {uploaded_file.name}")
        except Exception as e:
            st.error(f"Failed to read uploaded file: {e}")

# --- Create Tabs ---
if st.session_state.df is not None:
    tab1, tab2, tab3 = st.tabs(["Upload & Preview", "Data Cleaning", "Visualizations & ML"])
    
    # ============ TAB 1: UPLOAD & PREVIEW ============
    with tab1:
        st.subheader("Dataset Preview")
        st.markdown("<div style='margin-top:6px'><small style='color:#6B7280'>Preview the uploaded file before cleaning</small></div>", unsafe_allow_html=True)
        st.write(f"**File name:** {st.session_state.uploaded_filename}")
        st.write(f"**Dataset shape:** {st.session_state.df.shape[0]} rows × {st.session_state.df.shape[1]} columns")
        
        st.markdown("#### Raw Data")
        st.dataframe(st.session_state.df.head(20), use_container_width=True)
        
        st.markdown("#### Dataset Information")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Rows", st.session_state.df.shape[0])
        with col2:
            st.metric("Total Columns", st.session_state.df.shape[1])
        with col3:
            st.metric("Data Types", len(st.session_state.df.dtypes.unique()))
        
        st.markdown("#### Basic Statistics")
        st.write(st.session_state.df.describe())
    
    # ============ TAB 2: DATA CLEANING ============
    with tab2:
        st.subheader("Data Cleaning & Transformation")
        st.markdown("<div style='margin-top:6px'><small style='color:#6B7280'>Run cleaning on the dataset in this tab</small></div>", unsafe_allow_html=True)
        
        if st.button("Start Cleaning Process", type="primary", use_container_width=True):
            with st.status("AI Sculptor at work", expanded=True) as status:
                st.write("Analyzing data structures")
                # Use the cached method for processing
                st.session_state.results = perform_ai_sculpting(st.session_state.df)
                status.update(label="Sculpting Complete", state="complete", expanded=False)
        
        # Show results if cleaning has been done
        if st.session_state.results is not None:
            results = st.session_state.results
            df_cleaned = results["df_cleaned"]
            quality_score = results["quality_score"]
            quality_grade = results["quality_grade"]
            duplicates = results["duplicates"]
            total_outliers = results["total_outliers"]
            insights = results["insights"]
            invalid_dates = results["invalid_dates"]
            missing = results["missing"]

            st.markdown("---")
            st.subheader("Executive Summary")
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Quality Score", f"{quality_score}/100", delta=quality_grade)
            with col2:
                st.metric("Data Volume", f"{df_cleaned.shape[0]} rows", delta=f"-{duplicates} dups")
            with col3:
                st.metric("Dimensions", f"{df_cleaned.shape[1]} cols")
            with col4:
                st.metric("Anomalies", total_outliers, delta="Detected", delta_color="inverse")

            # --- Export Section ---
            st.markdown("### Export Sculpted Data")
            csv = df_cleaned.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="Download Cleaned CSV ⬇️",
                data=csv,
                file_name=f"sculpted_{st.session_state.uploaded_filename}",
                mime='text/csv',
                use_container_width=False
            )

            # --- Cleaned Data & Comparison ---
            st.markdown("---")
            tab_preview, tab_compare = st.tabs(["Cleaned Dataset", "Before vs After"])
            
            with tab_preview:
                st.dataframe(df_cleaned.head(20), use_container_width=True)
            
            with tab_compare:
                col_a, col_b = st.columns(2)
                col_a.markdown("#### Original (Raw)")
                col_a.write(st.session_state.df.describe())
                col_b.markdown("#### Cleaned")
                col_b.write(df_cleaned.describe())

            # --- Insights & Issues ---
            st.markdown("---")
            row2_1, row2_2 = st.columns(2)
            
            with row2_1:
                st.subheader("AI Intelligence Report")
                for insight in insights:
                    st.info(insight)

            with row2_2:
                st.subheader("Transformation Log")
                st.json({
                    "Missing Values Imputed": sum(missing.values()),
                    "Duplicates Purged": duplicates,
                    "Date Formats Standardized": invalid_dates,
                    "Outliers Isolated": total_outliers
                })
        else:
            st.info("Click the 'Start Cleaning Process' button to clean your data")
    
    # ============ TAB 3: VISUALIZATIONS & ML ============
    with tab3:
        if st.session_state.results is not None:
            st.subheader("Visualizations & Machine Learning")
            st.markdown("<div style='margin-top:6px'><small style='color:#6B7280'>Explore charts and train models</small></div>", unsafe_allow_html=True)
            
            df_cleaned = st.session_state.results["df_cleaned"]

            # --- Top-row Diagnostics: Forecasting | Anomaly Detection | Automated Decisions ---
            fc_col, an_col, dec_col = st.columns(3)

            # Predictive Forecasting Column
            with fc_col:
                st.markdown("#### Predictive Forecasting")
                st.markdown("Basic trend forecasting for a numeric column over a datetime index.")
                datetime_cols_local = [c for c in df_cleaned.columns if pd.api.types.is_datetime64_any_dtype(df_cleaned[c])]
                if len(datetime_cols_local) == 0:
                    for c in df_cleaned.select_dtypes(include=['object']).columns:
                        try:
                            parsed = pd.to_datetime(df_cleaned[c], errors='coerce')
                            if parsed.notna().sum() > 0:
                                datetime_cols_local.append(c)
                                df_cleaned[c] = parsed
                                break
                        except Exception:
                            continue

                if len(datetime_cols_local) > 0 and len(df_cleaned.select_dtypes(include=['number']).columns) > 0:
                    date_col_local = st.selectbox("Date column", datetime_cols_local, key='date_col_fc')
                    fc_target_local = st.selectbox("Forecast target (numeric)", df_cleaned.select_dtypes(include=['number']).columns, key='fc_target_fc')
                    periods_local = st.number_input("Periods to forecast (steps)", min_value=1, max_value=24, value=6, key='fc_periods')
                    if st.button("Run Forecast", key='run_fc_col'):
                        try:
                            ts = df_cleaned[[date_col_local, fc_target_local]].dropna()
                            ts = ts.sort_values(date_col_local).set_index(date_col_local).resample('D').mean().interpolate()
                            idx = np.arange(len(ts))
                            vals = ts[fc_target_local].values
                            coef = np.polyfit(idx, vals, 1)
                            future_idx = np.arange(len(ts), len(ts) + periods_local)
                            forecast_vals = np.polyval(coef, future_idx)
                            forecast_index = pd.date_range(start=ts.index[-1] + pd.Timedelta(days=1), periods=periods_local, freq='D')
                            fc_df = pd.DataFrame({fc_target_local: forecast_vals}, index=forecast_index)
                            st.write(fc_df)
                            st.caption("Explanation: Simple linear-trend forecast. For robust forecasting use ARIMA/Prophet or ML time-series models.")
                        except Exception as e:
                            st.error(f"Forecasting failed: {e}")
                else:
                    st.info("No suitable datetime + numeric columns found for forecasting.")

            # Anomaly Detection Column
            with an_col:
                st.markdown("#### Anomaly Detection")
                st.markdown("Quick anomaly detection using IQR and z-score methods.")
                numeric_cols_local = df_cleaned.select_dtypes(include=['number']).columns.tolist()
                if len(numeric_cols_local) > 0:
                    anomaly_col_local = st.selectbox("Select column to detect anomalies", numeric_cols_local, key='anom_col_col')
                    z_thresh_local = st.slider("Z-score threshold", 2.0, 5.0, 3.0, step=0.5, key='zth_col')
                    if st.button("Detect Anomalies", key='run_anom_col'):
                        col_series = df_cleaned[anomaly_col_local].dropna()
                        mean = col_series.mean()
                        std = col_series.std()
                        z_scores = (col_series - mean) / (std if std != 0 else 1)
                        anom_z = col_series[np.abs(z_scores) > z_thresh_local]
                        q1 = col_series.quantile(0.25)
                        q3 = col_series.quantile(0.75)
                        iqr = q3 - q1
                        lower = q1 - 1.5 * iqr
                        upper = q3 + 1.5 * iqr
                        anom_iqr = col_series[(col_series < lower) | (col_series > upper)]
                        st.write(f"Z-score anomalies: {len(anom_z)} samples")
                        if len(anom_z) > 0:
                            st.dataframe(anom_z.head(10))
                        st.write(f"IQR anomalies: {len(anom_iqr)} samples")
                        if len(anom_iqr) > 0:
                            st.dataframe(anom_iqr.head(10))
                        st.caption("Explanation: Z-score flags values far from the mean; IQR is robust to outliers. Review flagged samples.")
                else:
                    st.info("No numeric columns available for anomaly detection.")

            # Automated Decision-Making Column
            with dec_col:
                st.markdown("#### Automated Decision-Making")
                st.markdown("Actionable recommendations based on data quality, anomalies, and model signals.")
                try:
                    qs_local = st.session_state.results.get('quality_score', None) if st.session_state.results else None
                except Exception:
                    qs_local = None
                recs = []
                if qs_local is not None:
                    if qs_local < 50:
                        recs.append("High priority: Significant data quality issues — strong cleaning recommended.")
                    elif qs_local < 75:
                        recs.append("Medium priority: Review missing values and format inconsistencies.")
                    else:
                        recs.append("Low priority: Data ready for modeling; validate model performance.")
                try:
                    tot_out_local = st.session_state.results.get('total_outliers', 0)
                    if tot_out_local and tot_out_local > 0:
                        recs.append(f"Detected {tot_out_local} outliers — consider capping/imputing or exclusion for training.")
                except Exception:
                    pass
                if len(recs) == 0:
                    recs.append("No automated recommendations available — run cleaning and diagnostics first.")
                for r in recs:
                    st.info(r)
                st.caption("Explanation: These recommendations are heuristics to prioritize actions; validate on a holdout sample.")
            
            # Visualizations
            st.markdown("#### Data Visualizations")
            tab_viz1, tab_viz2, tab_viz3 = st.tabs(["Correlation Heatmap", "Trends", "Distributions"])
            
            with tab_viz1:
                heatmap_file = save_heatmap(df_cleaned)
                if heatmap_file:
                    img_path = os.path.join(BASE_DIR, 'static/images', heatmap_file)
                    if os.path.exists(img_path):
                        st.image(img_path, caption="Feature Correlation Heatmap", use_container_width=True)
                else:
                    st.info("Not enough numeric data for a heatmap.")

            numeric_cols = df_cleaned.select_dtypes(include=['number']).columns
            
            with tab_viz2:
                if len(numeric_cols) > 0:
                    col_trend = st.selectbox("Select Column for Trend", numeric_cols, key='trend')
                    trend_file = save_trend_plot(df_cleaned, col_trend)
                    img_path = os.path.join(BASE_DIR, 'static/images', trend_file)
                    if os.path.exists(img_path):
                        st.image(img_path, use_container_width=True)
                else:
                    st.info("No numeric columns available for trend analysis.")
            
            with tab_viz3:
                if len(numeric_cols) > 0:
                    col_dist = st.selectbox("Select Column for Distribution", numeric_cols, key='dist')
                    dist_file = save_distribution_plot(df_cleaned, col_dist)
                    img_path = os.path.join(BASE_DIR, 'static/images', dist_file)
                    if os.path.exists(img_path):
                        st.image(img_path, use_container_width=True)
                else:
                    st.info("No numeric columns available for distribution analysis.")

            # Machine Learning
            st.markdown("---")
            st.markdown("#### Automated Machine Learning")
            st.info("Select a target column you want to predict. The AI will automatically choose the best model.")
            
            target_col = st.selectbox("Select Target Column", df_cleaned.columns)
            if st.button("Train AI Model", type="primary", use_container_width=False):
                try:
                    with st.spinner("Training model..."):
                        task_type, score = train_model(df_cleaned, target_col)
                        st.success(f"Model Trained Successfully! (Task: {task_type})")
                        st.metric("Model Accuracy / Score", f"{score:.2%}")
                except Exception as e:
                    st.error(f"Error during model training: {e}")

            # --- Advanced AI Insights (Regression, Forecasting, Anomaly Detection, Decisions, Stats) ---
            st.markdown("---")
            st.subheader("Advanced AI Insights")

            # Regression Analysis
            st.markdown("#### Regression Analysis")
            st.markdown("Select a numeric target and features to run a quick linear regression and get an explanation of the results.")
            numeric_cols = df_cleaned.select_dtypes(include=['number']).columns.tolist()
            if len(numeric_cols) >= 2:
                reg_target = st.selectbox("Regression target", numeric_cols, key='reg_target')
                reg_features = st.multiselect("Regression features (numeric)", [c for c in numeric_cols if c != reg_target], default=[c for c in numeric_cols if c != reg_target][:2], key='reg_feats')
                if st.button("Run Regression", key='run_reg'):
                    try:
                        from sklearn.linear_model import LinearRegression
                        X = df_cleaned[reg_features].dropna()
                        y = df_cleaned.loc[X.index, reg_target]
                        model = LinearRegression().fit(X.values, y.values)
                        r2 = model.score(X.values, y.values)
                        coefs = dict(zip(reg_features, model.coef_))
                        st.metric("R²", f"{r2:.3f}")
                        st.write("**Coefficients**")
                        st.write(coefs)
                        st.caption("Explanation: R² measures how much variance in the target is explained by the features. Coefficients show the expected change in target per unit change in each predictor, holding others constant.")
                    except Exception:
                        # Fallback simple correlation-based explanation
                        st.warning("Unable to run sklearn linear regression — falling back to correlation summary.")
                        corrs = df_cleaned[[reg_target] + reg_features].corr()[reg_target].drop(reg_target)
                        st.write(corrs.sort_values(ascending=False))
                        st.caption("Explanation: Correlations indicate linear association strength between each feature and the target.")
            else:
                st.info("Need at least two numeric columns for regression analysis.")

            

            # Statistical Evaluations
            st.markdown("#### Statistical Evaluations")
            st.markdown("Provides quick statistical metrics (skewness, kurtosis, correlation) and short interpretation comments.")
            if len(numeric_cols) > 0:
                stats_df = pd.DataFrame(index=numeric_cols)
                stats_df['mean'] = df_cleaned[numeric_cols].mean()
                stats_df['std'] = df_cleaned[numeric_cols].std()
                stats_df['skew'] = df_cleaned[numeric_cols].skew()
                stats_df['kurtosis'] = df_cleaned[numeric_cols].kurtosis()
                st.dataframe(stats_df)
                corr = df_cleaned[numeric_cols].corr()
                st.markdown("**Correlation matrix**")
                st.dataframe(corr)
                st.caption("Explanation: Skewness > |1| indicates strong asymmetry. High kurtosis indicates heavy tails. Use correlations to detect multicollinearity and strong predictor signals.")
            else:
                st.info("No numeric columns for statistical evaluations.")
        else:
            st.info("👈 Please clean your data first in the 'Data Cleaning' tab to see visualizations and train models.")
else:
    st.markdown("""
    <div style='text-align: center; margin-top: 50px;'>
        <h2>Welcome to Data Sculptor</h2>
        <p>Please upload a CSV file from the sidebar to get started.</p>
    </div>
    """, unsafe_allow_html=True)