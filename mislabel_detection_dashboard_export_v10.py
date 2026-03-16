# -*- coding: utf-8 -*-
"""
Enhanced Mislabel Detection with Cross-Type Mislabeling Suggestion
- Original functionality: detect anomalies within each part type
- NEW: Suggest which OTHER part type a flagged part might be mislabeled as
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st

st.set_page_config(page_title="Mislabel Detection (Enhanced)", layout="wide")

st.title("Manufacturing Mislabel Detection (Enhanced)")

# ===== CONTACT INFO =====
st.sidebar.divider()
st.sidebar.info(
    "**Questions or Issues?**\n\n"
    "📧 Contact: [chad@hertzler.com](mailto:chad@hertzler.com)"
)
st.sidebar.divider()

# ===== HELP SECTION =====
with st.expander("❓ How This App Works (Click to Expand)", expanded=False):
    st.markdown("""
    ### What This App Does
    This app detects **anomalous parts** within each declared part type using multivariate anomaly detection.
    It then suggests which **other part types** a flagged part might actually be (mislabeling detection).
    
    ### Three-Step Workflow
    
    **Step 1: Map Your Columns**
    - Select which column contains Part IDs (unique identifier for each part)
    - Select which column contains Part Numbers/Labels (the declared type: PN-A, PN-B, etc.)
    - Select 3+ numeric features (measurements, parameters, etc.) to analyze
    - The app shows data quality so you can pick high-completeness columns
    
    **Step 2: Run Analysis**
    - Click "Analyze Data" to detect anomalies
    - The app automatically determines how many parts should be flagged as anomalous (5-15% typically)
    - For each part type, anomalies are detected independently
    
    **Step 3: Explore Results**
    - See which parts are flagged as anomalous
    - For each anomalous part, see **mislabeling suggestions** - which other part type it might actually be
    - View diagnostics, distributions, and correlations
    
    ### Key Features
    
    🤖 **Smart Contamination Detection** - Automatically decides what % of parts should be flagged based on your data
    
    🔍 **Cross-Type Mislabeling** - Shows likelihood that a flagged part actually belongs to a different part type
    
    📊 **Rich Visualizations** - Distributions, PCA, correlations, feature importance
    
    📥 **Export Results** - Download complete analysis or just the anomalies
    
    ### Example
    ```
    You declare a part as PN-A, but:
    - Measurement 1 is 8.5 (PN-A typically 10.0)
    - Measurement 2 is 6.3 (PN-A typically 6.0)
    
    The app flags it as anomalous WITHIN PN-A.
    Then it checks: "Does this fit PN-B better?" 
    If measurements match PN-B's pattern, it suggests: 
    ⚠️ POSSIBLE MISLABEL: likely PN-B not PN-A (88% likelihood)
    ```
    
    ### Tips
    - Start with **Demo Data** to understand how it works
    - Select features with **>90% data completeness** (shown in Step 1)
    - Anomalies detected **per part type** - each type analyzed separately
    - Download results to investigate in Excel
    """)

st.divider()


# ===== DYNAMIC CONTAMINATION DETECTION =====
def estimate_optimal_contamination_per_type(anomaly_scores_dict: dict) -> tuple[dict, str]:
    """Intelligently estimate optimal contamination rates per part type."""
    contamination_map = {}
    explanations = []
    
    for part_type, scores in anomaly_scores_dict.items():
        scores_clean = np.array(scores)[np.isfinite(scores)]
        
        if len(scores_clean) < 5:
            contamination_map[part_type] = 0.15
            explanations.append(f"{part_type}: <5 samples → 15% (default)")
            continue
        
        sorted_scores = np.sort(scores_clean)
        gaps = np.diff(sorted_scores)
        
        if len(gaps) > 0:
            largest_gap_idx = np.argmax(gaps)
            largest_gap_value = gaps[largest_gap_idx]
            gap_percentile = (largest_gap_idx + 1) / len(scores_clean)
            median_gap = np.median(gaps)
            
            if largest_gap_value > 0.5 * median_gap and 0.05 <= gap_percentile <= 0.30:
                contamination_map[part_type] = gap_percentile
                explanations.append(f"{part_type}: Elbow at {gap_percentile*100:.1f}%")
                continue
        
        mean_score = np.mean(scores_clean)
        std_score = np.std(scores_clean)
        threshold_z = mean_score + 1.5 * std_score
        n_anomalies = np.sum(scores_clean > threshold_z)
        contamination_z = max(0.05, min(0.25, n_anomalies / len(scores_clean)))
        
        if 0.05 <= contamination_z <= 0.25:
            contamination_map[part_type] = contamination_z
            explanations.append(f"{part_type}: Z-score {contamination_z*100:.1f}%")
            continue
        
        q1, q3 = np.percentile(scores_clean, [25, 75])
        iqr = q3 - q1
        outliers = (scores_clean < q1 - 1.5 * iqr) | (scores_clean > q3 + 1.5 * iqr)
        n_outliers = np.sum(outliers)
        contamination_iqr = max(0.05, min(0.25, n_outliers / len(scores_clean)))
        
        if 0.05 <= contamination_iqr <= 0.25:
            contamination_map[part_type] = contamination_iqr
            explanations.append(f"{part_type}: IQR {contamination_iqr*100:.1f}%")
            continue
        
        if len(scores_clean) < 20:
            cont = 0.15
        elif len(scores_clean) < 100:
            cont = 0.10
        else:
            cont = 0.08
        
        contamination_map[part_type] = cont
        explanations.append(f"{part_type}: Fallback ({len(scores_clean)} samples) → {cont*100:.0f}%")
    
    explanation_text = "\n".join(explanations) if explanations else "No part types found"
    return contamination_map, explanation_text


# ===== NEW: CROSS-TYPE MISLABELING DETECTION =====
def detect_cross_type_mislabeling(df: pd.DataFrame, num_col: str, feat_cols: list[str], anomalous_indices: list) -> dict:
    """
    For each anomalous part, calculate how well it fits each other part type.
    Returns: {anomalous_idx: {other_type: likelihood_score, ...}}
    """
    all_part_types = sorted(df[num_col].unique())
    results = {}
    
    for anom_idx in anomalous_indices:
        if anom_idx >= len(df):
            continue
        
        declared_type = df.iloc[anom_idx][num_col]
        
        # Get features and safely convert to numeric
        part_features_raw = df.iloc[anom_idx][feat_cols].values
        part_features = []
        has_nan = False
        
        for val in part_features_raw:
            try:
                numeric_val = float(val)
                if pd.isna(numeric_val) or np.isnan(numeric_val):
                    has_nan = True
                part_features.append(numeric_val)
            except (ValueError, TypeError):
                has_nan = True
                part_features.append(np.nan)
        
        part_features = np.array(part_features)
        
        # Skip if any missing features
        if has_nan or np.any(np.isnan(part_features)):
            continue
        
        type_likelihoods = {}
        
        # Check fit against each OTHER part type
        for other_type in all_part_types:
            if other_type == declared_type:
                continue
            
            # Get baseline stats for this other type - safely convert to float
            other_type_mask = df[num_col] == other_type
            if not other_type_mask.any():
                continue
            
            other_type_data = df.loc[other_type_mask, feat_cols].copy()
            
            # Convert all to numeric
            for col in feat_cols:
                other_type_data[col] = pd.to_numeric(other_type_data[col], errors='coerce')
            
            # Drop rows with any NaN
            other_type_data = other_type_data.dropna()
            
            if len(other_type_data) < 2:
                continue
            
            means = other_type_data.mean()
            stds = other_type_data.std()
            
            # Calculate z-scores relative to other type
            z_scores = []
            for feat, val in zip(feat_cols, part_features):
                if pd.notna(val) and pd.notna(stds[feat]) and stds[feat] > 0:
                    z = (val - means[feat]) / stds[feat]
                    z_scores.append(abs(z))
                elif pd.notna(val) and pd.notna(means[feat]):
                    z_scores.append(0 if val == means[feat] else 999)
                else:
                    z_scores.append(999)
            
            # Calculate fit score: lower avg z-score = better fit
            valid_z = [z for z in z_scores if z < 999]
            if not valid_z:
                continue
            
            avg_z = np.mean(valid_z)
            
            # Convert to likelihood (0-100, higher = more likely to be this type)
            # Low z-scores (< 2) indicate good fit
            likelihood = max(0, 100 - (avg_z * 15))  # Tuning factor: 15
            
            type_likelihoods[other_type] = {
                "likelihood": likelihood,
                "avg_z_score": avg_z,
                "z_scores_by_feature": dict(zip(feat_cols, [f"{z:.2f}" for z in z_scores]))
            }
        
        if type_likelihoods:
            results[anom_idx] = type_likelihoods
    
    return results


@st.cache_data
def generate_demo_data():
    np.random.seed(42)
    records = []
    for i in range(500):
        ptype = np.random.choice(["PN-A", "PN-B", "PN-C"])
        if ptype == "PN-A":
            m1, m2 = np.random.normal(10.0, 0.05), np.random.normal(6.0, 0.02)
            t, p, s = np.random.normal(200, 2), np.random.normal(5.0, 0.2), np.random.normal(100, 5)
        elif ptype == "PN-B":
            m1, m2 = np.random.normal(9.5, 0.03), np.random.normal(6.5, 0.03)
            t, p, s = np.random.normal(190, 2), np.random.normal(4.5, 0.2), np.random.normal(95, 5)
        else:
            m1, m2 = np.random.normal(10.2, 0.04), np.random.normal(5.8, 0.01)
            t, p, s = np.random.normal(210, 2), np.random.normal(5.5, 0.2), np.random.normal(105, 5)
        records.append({
            "PartID": f"P{i+1:03d}", "DeclaredPartNo": ptype, "Meas1": round(m1, 4), "Meas2": round(m2, 4),
            "Temperature": round(t, 2), "Pressure": round(p, 3), "Speed": round(s, 1),
            "Machine": np.random.choice(["M1", "M2"]), "Tool": np.random.choice(["T1", "T2", "T3"]),
            "Operator": np.random.choice(["O123", "O124", "O125"])
        })
    return pd.DataFrame(records)

# ===== INITIALIZE SESSION STATE =====
if "analysis_complete" not in st.session_state:
    st.session_state.analysis_complete = False
    st.session_state.df_analysis = None
    st.session_state.id_col = None
    st.session_state.num_col = None
    st.session_state.feat_cols = None
    st.session_state.contamination_map = {}
    st.session_state.mislabeling_suggestions = {}

# ===== DATA LOADING =====
st.sidebar.header("Data Source")
data_source = st.sidebar.radio(
    "Choose:",
    ["Demo Data", "Upload CSV"],
    help="Use demo data to test the app, or upload your own CSV file."
)

if data_source == "Upload CSV":
    uploaded_file = st.file_uploader("Upload CSV", type=["csv"])
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.success(f"✓ Loaded {len(df)} rows")
    else:
        st.info("Upload a CSV file to begin")
        st.stop()
else:
    df = generate_demo_data()
    st.success("✓ Demo data loaded")

# ===== COLUMN MAPPING =====
st.header("Step 1: Map Your Columns")

cols = df.columns.tolist()

# Calculate data completeness for each column
col_completeness = []
for col in cols:
    non_null = df[col].notna().sum()
    pct = (non_null / len(df)) * 100
    col_completeness.append({"Column": col, "Non-Null": non_null, "Complete %": pct})

completeness_df = pd.DataFrame(col_completeness).sort_values("Complete %", ascending=False)

# Show column quality info
with st.expander("📊 Column Quality Analysis", expanded=True):
    st.write("**⚠️ Select features with HIGH data completeness to avoid losing data**")
    st.dataframe(completeness_df, use_container_width=True)

col1, col2 = st.columns(2)

with col1:
    id_opts = ["(Auto-Generate ID)"] + cols
    id_col_input = st.selectbox(
        "Part ID Column",
        id_opts,
        help="Column that uniquely identifies each part/record"
    )
    id_col = None if id_col_input == "(Auto-Generate ID)" else id_col_input
    
    num_col = st.selectbox(
        "Part Number/Label Column",
        cols,
        help="Column with part type/SKU labels (e.g., PN-A, PN-B). Anomalies detected per type."
    )

with col2:
    good_cols = completeness_df[completeness_df["Complete %"] > 90]["Column"].tolist()
    
    st.write(f"**Recommended features ({len(good_cols)}):** Columns with >90% data")
    
    feat_cols = st.multiselect(
        "Select 3+ Features (choose columns with high completeness)",
        cols,
        default=good_cols[:min(5, len(good_cols))],
        help="These numeric features will be analyzed for anomalies within each part type."
    )
    
    selected_completeness = completeness_df[completeness_df["Column"].isin(feat_cols)]
    low_quality = selected_completeness[selected_completeness["Complete %"] < 80]
    
    if len(low_quality) > 0:
        st.warning(f"⚠️ {len(low_quality)} selected feature(s) have <80% data - may lose many rows!")
        st.dataframe(low_quality, use_container_width=True)

# ===== ANALYSIS BUTTON =====
st.header("Step 2: Run Analysis")

analysis_ready = st.button("▶ Analyze Data", type="primary", use_container_width=True)

if analysis_ready:
    if len(feat_cols) < 3:
        st.error("Select at least 3 features")
        st.stop()
    if not num_col:
        st.error("Select a Part Number column")
        st.stop()
    
    st.session_state.analysis_complete = True
    st.session_state.id_col = id_col
    st.session_state.num_col = num_col
    st.session_state.feat_cols = feat_cols
    st.session_state.df_analysis = None

# Check if analysis has been run
if not st.session_state.get("analysis_complete", False):
    st.stop()

# ===== DO ANALYSIS =====
if st.session_state.df_analysis is None:
    id_col = st.session_state.id_col
    num_col = st.session_state.num_col
    feat_cols = st.session_state.feat_cols
    
    df_analysis = df.copy()
    
    if id_col is None:
        df_analysis.insert(0, "_PartID", [f"Row_{i+1}" for i in range(len(df_analysis))])
        id_col = "_PartID"
    
    # ===== DIAGNOSTIC INFO =====
    with st.expander("🔍 Diagnostic Info - Data Quality Check", expanded=False):
        st.write("**Feature columns before conversion:**")
        diag_before = pd.DataFrame({
            "Feature": feat_cols,
            "Type": [str(df_analysis[col].dtype) for col in feat_cols],
            "Non-Null": [df_analysis[col].notna().sum() for col in feat_cols],
            "Null Count": [df_analysis[col].isna().sum() for col in feat_cols],
            "Sample": [str(df_analysis[col].iloc[0])[:50] if len(df_analysis) > 0 else "N/A" for col in feat_cols]
        })
        st.dataframe(diag_before, use_container_width=True)
        
        # Check for ANY missing values
        missing_per_col = {col: df_analysis[col].isna().sum() for col in feat_cols}
        total_missing = sum(missing_per_col.values())
        st.write(f"**Total missing values across all features:** {total_missing}")
    
    # Convert feature columns to numeric
    for col in feat_cols:
        df_analysis[col] = pd.to_numeric(df_analysis[col], errors='coerce')
    
    with st.expander("🔍 After Numeric Conversion", expanded=False):
        st.write("**After pd.to_numeric() - This is what gets analyzed:**")
        diag_after = pd.DataFrame({
            "Feature": feat_cols,
            "Type": [str(df_analysis[col].dtype) for col in feat_cols],
            "Non-Null": [df_analysis[col].notna().sum() for col in feat_cols],
            "NaN Count": [df_analysis[col].isna().sum() for col in feat_cols],
            "Sample": [str(df_analysis[col].iloc[0])[:50] if len(df_analysis) > 0 else "N/A" for col in feat_cols]
        })
        st.dataframe(diag_after, use_container_width=True)
    
    # Drop rows with missing values in features
    rows_before = len(df_analysis)
    df_analysis = df_analysis.dropna(subset=feat_cols)
    rows_after = len(df_analysis)
    rows_dropped = rows_before - rows_after
    
    if rows_dropped > 0:
        st.warning(f"⚠️ Removed {rows_dropped} rows with missing values in features ({(rows_dropped/rows_before)*100:.1f}%)")
    else:
        st.success("✓ All features have valid data - no rows dropped")
    
    if len(df_analysis) == 0:
        st.error("❌ No valid data after removing missing values.")
        st.error("**Why this happened:**")
        st.write("- The selected columns may contain values that couldn't be converted to numbers (e.g., text, symbols)")
        st.write("- There may be hidden missing values (spaces, empty strings, etc.)")
        st.write("- Try deselecting some columns and clicking 'Analyze Data' again")
        st.stop()
    
    st.info(f"✓ Working with {rows_after} rows ({(rows_after/rows_before)*100:.1f}% of original data)")
    
    # ===== DYNAMIC CONTAMINATION DETECTION =====
    st.subheader("🤖 Smart Contamination Detection")
    
    anomaly_scores_by_type = {}
    
    for part in df_analysis[num_col].unique():
        idx = df_analysis[df_analysis[num_col] == part].index
        subset = df_analysis.loc[idx, feat_cols].copy()
        
        # Safely convert to numeric
        for col in feat_cols:
            subset[col] = pd.to_numeric(subset[col], errors='coerce')
        
        # Drop rows with NaN
        subset = subset.dropna()
        
        if len(subset) > 1:
            model_initial = IsolationForest(contamination=0.1, random_state=42)
            decision_scores = model_initial.fit_predict(subset)
            anomaly_scores_by_type[part] = -model_initial.decision_function(subset)
    
    contamination_map, contamination_explanation = estimate_optimal_contamination_per_type(anomaly_scores_by_type)
    
    st.session_state.contamination_map = contamination_map
    
    st.info(
        f"**Automatically Detected Contamination Rates:**\n\n"
        f"{contamination_explanation}"
    )
    
    # ===== ANOMALY DETECTION =====
    st.subheader("Running Anomaly Detection...")
    
    df_analysis["AnomalyFlag"] = "Normal"
    df_analysis["AnomalyScore"] = 0.0
    anomalous_indices = []
    
    for part in df_analysis[num_col].unique():
        idx = df_analysis[df_analysis[num_col] == part].index
        subset = df_analysis.loc[idx, feat_cols].copy()
        
        # Safely convert to numeric
        for col in feat_cols:
            subset[col] = pd.to_numeric(subset[col], errors='coerce')
        
        # Drop rows with NaN
        subset = subset.dropna()
        
        if len(subset) < 2:
            st.warning(f"⚠️ Part type '{part}' has only {len(subset)} sample(s) - cannot detect anomalies")
            continue
        
        cont_rate = contamination_map.get(part, 0.10)
        
        model = IsolationForest(contamination=cont_rate, random_state=42)
        preds = model.fit_predict(subset)
        decision_scores = model.decision_function(subset)
        
        df_analysis.loc[idx, "AnomalyFlag"] = ["Anomaly" if p == -1 else "Normal" for p in preds]
        df_analysis.loc[idx, "AnomalyScore"] = -decision_scores
        
        # Track anomalous indices for cross-type analysis
        anomalous_indices.extend(idx[preds == -1].tolist())
    
    # ===== NEW: CROSS-TYPE MISLABELING DETECTION =====
    st.subheader("🔍 Cross-Type Mislabeling Analysis...")
    
    mislabeling_suggestions = detect_cross_type_mislabeling(df_analysis, num_col, feat_cols, anomalous_indices)
    st.session_state.mislabeling_suggestions = mislabeling_suggestions
    
    st.success(f"✓ Analyzed {len(anomalous_indices)} anomalies for possible mislabeling")
    
    st.session_state.df_analysis = df_analysis
    st.session_state.id_col = id_col
else:
    df_analysis = st.session_state.df_analysis
    id_col = st.session_state.id_col
    num_col = st.session_state.num_col
    feat_cols = st.session_state.feat_cols
    contamination_map = st.session_state.contamination_map
    mislabeling_suggestions = st.session_state.mislabeling_suggestions

# ===== EXPORT =====
st.sidebar.header("Export")
suspects = df_analysis[df_analysis["AnomalyFlag"] == "Anomaly"]

col_e1, col_e2 = st.sidebar.columns(2)
with col_e1:
    st.download_button(
        "📥 All Data",
        df_analysis.to_csv(index=False),
        "data.csv",
        "text/csv",
        help="Download complete analysis results"
    )
with col_e2:
    st.download_button(
        "📥 Anomalies",
        suspects.to_csv(index=False),
        "anomalies.csv",
        "text/csv",
        help="Download only flagged anomalies"
    )

if st.sidebar.button("🔄 Reset", use_container_width=True, help="Clear analysis and start over"):
    st.session_state.analysis_complete = False
    st.session_state.df_analysis = None
    st.rerun()

# ===== RESULTS =====
st.header("Results")

show_anom = st.checkbox("Show only anomalies", value=False)
display_df = suspects if show_anom else df_analysis

st.subheader(f"Data ({len(display_df)} rows)")
if len(display_df) == 0:
    st.error("❌ No anomalies found")
else:
    st.dataframe(display_df, use_container_width=True, height=500)

# ===== ANOMALY SCORE DISTRIBUTION =====
if len(df_analysis) > 0:
    st.subheader("Anomaly Score Distribution")
    
    fig, axes = plt.subplots(1, len(contamination_map), figsize=(5 * len(contamination_map), 4))
    if len(contamination_map) == 1:
        axes = [axes]
    
    for idx, (part_type, ax) in enumerate(zip(sorted(contamination_map.keys()), axes)):
        part_data = df_analysis[df_analysis[num_col] == part_type]
        if len(part_data) > 0:
            ax.hist(part_data["AnomalyScore"], bins=20, alpha=0.7, edgecolor='black')
            threshold = np.percentile(part_data["AnomalyScore"], (1 - contamination_map[part_type]) * 100)
            ax.axvline(threshold, color='red', linestyle='--', linewidth=2, label=f'Threshold ({contamination_map[part_type]*100:.0f}%)')
            ax.set_title(f"{part_type} (n={len(part_data)})")
            ax.set_xlabel("Anomaly Score")
            ax.set_ylabel("Count")
            ax.legend()
    
    try:
        plt.tight_layout()
    except:
        pass  # tight_layout failed, matplotlib will handle it
    st.pyplot(fig)
    plt.close(fig)

# ===== EXPLAIN ROW (WITH MISLABELING SUGGESTION) =====
st.subheader("Explain a Row (With Mislabeling Detection)")
part_ids = display_df[id_col].unique().tolist() if len(display_df) > 0 else []
if part_ids:
    selected = st.selectbox("Select Part ID", part_ids, help="Pick a part to see why it was flagged (or not)")
    try:
        row = df_analysis[df_analysis[id_col] == selected].iloc[0]
        row_idx = df_analysis[df_analysis[id_col] == selected].index[0]
        
        st.write(f"**ID:** {selected} | **Type:** {row[num_col]} | **Flag:** {row['AnomalyFlag']} | **Score:** {row['AnomalyScore']:.3f}")
        
        base = df_analysis[df_analysis[num_col] == row[num_col]]
        means = base[feat_cols].mean()
        stds = base[feat_cols].std(ddof=1)
        
        vals = pd.to_numeric(row[feat_cols], errors='coerce')
        
        z_scores = []
        for feat in feat_cols:
            if pd.notna(stds[feat]) and stds[feat] > 0:
                z = (vals[feat] - means[feat]) / stds[feat]
            else:
                z = 0.0 if vals[feat] == means[feat] else np.inf
            z_scores.append(z)
        
        z = np.array(z_scores)
        
        exp_df = pd.DataFrame({
            "Feature": feat_cols,
            "Value": vals.values,
            "Mean": means.values,
            "Std": stds.values,
            "|Z|": np.abs(z)
        }).sort_values("|Z|", ascending=False)
        
        st.subheader("Feature Z-Scores (vs. Declared Type)")
        st.dataframe(exp_df.style.format({c: "{:.3f}" for c in exp_df.columns[1:]}), use_container_width=True)
        
        fig, ax = plt.subplots(figsize=(10, 4))
        valid = exp_df[np.isfinite(exp_df["|Z|"])]
        if len(valid) > 0:
            sns.barplot(data=valid, x="Feature", y="|Z|", hue="Feature", ax=ax, palette="husl", legend=False)
            ax.set_ylabel("|Z-score|")
            ax.set_title(f"Why {selected} is Anomalous in Type {row[num_col]}")
            plt.xticks(rotation=45, ha='right')
            try:
                plt.tight_layout()
            except:
                pass  # tight_layout failed, matplotlib will handle it
            st.pyplot(fig)
            plt.close(fig)
        
        # ===== MISLABELING SUGGESTION =====
        if row['AnomalyFlag'] == "Anomaly" and row_idx in mislabeling_suggestions:
            st.subheader("🚨 Possible Mislabeling Suggestion")
            
            suggestions = mislabeling_suggestions[row_idx]
            
            if suggestions:
                # Sort by likelihood descending
                sorted_suggestions = sorted(suggestions.items(), key=lambda x: x[1]['likelihood'], reverse=True)
                
                suggestion_df = pd.DataFrame([
                    {
                        "Possible Type": s[0],
                        "Likelihood (%)": round(s[1]['likelihood'], 1),
                        "Avg Z-Score": round(s[1]['avg_z_score'], 2),
                        "Fit Quality": "EXCELLENT" if s[1]['likelihood'] > 70 else "GOOD" if s[1]['likelihood'] > 50 else "FAIR" if s[1]['likelihood'] > 30 else "POOR"
                    }
                    for s in sorted_suggestions
                ])
                
                st.dataframe(suggestion_df, use_container_width=True)
                
                if len(sorted_suggestions) > 0:
                    best_match = sorted_suggestions[0]
                    if best_match[1]['likelihood'] > 50:
                        st.warning(
                            f"⚠️ **POSSIBLE MISLABEL:** This part appears to be **{best_match[0]}** "
                            f"(likelihood: {best_match[1]['likelihood']:.1f}%) rather than {row[num_col]}\n\n"
                            f"Z-Score breakdown when checked against {best_match[0]}:"
                        )
                        
                        z_breakdown = pd.DataFrame([
                            {"Feature": feat, "Z-Score": z_str}
                            for feat, z_str in best_match[1]['z_scores_by_feature'].items()
                        ])
                        st.dataframe(z_breakdown, use_container_width=True)
                    else:
                        st.info("Part is anomalous but doesn't strongly match any other type. May be defective.")
            else:
                st.info("No cross-type analysis available for this part.")
        else:
            st.info("This part is not flagged as anomalous, so no mislabeling check is needed.")
        
    except Exception as e:
        st.warning(f"Error: {e}")

# ===== SCATTER PLOT =====
if len(feat_cols) >= 2 and len(display_df) > 0:
    st.subheader("Scatter Plot")
    norm = display_df[display_df["AnomalyFlag"] == "Normal"].dropna(subset=[feat_cols[0], feat_cols[1]])
    anom = display_df[display_df["AnomalyFlag"] == "Anomaly"].dropna(subset=[feat_cols[0], feat_cols[1]])
    
    # For large datasets, sample normal points to avoid overplotting
    MAX_NORMAL_POINTS = 1000
    if len(norm) > MAX_NORMAL_POINTS:
        norm_sampled = norm.sample(n=MAX_NORMAL_POINTS, random_state=42)
        sample_info = f"(showing {MAX_NORMAL_POINTS:,} of {len(norm):,} normal points)"
    else:
        norm_sampled = norm
        sample_info = ""
    
    fig, ax = plt.subplots(figsize=(14, 7))
    
    if len(norm_sampled) > 0:
        sns.scatterplot(data=norm_sampled, x=feat_cols[0], y=feat_cols[1], hue=num_col, 
                      ax=ax, palette="Set2", s=10, alpha=0.7, edgecolor="white", linewidth=0.5)
    if len(anom) > 0:
        ax.scatter(anom[feat_cols[0]], anom[feat_cols[1]], color='red', s=10, marker='X', 
                  edgecolors='darkred', linewidth=2, alpha=0.95, label='ANOMALY', zorder=5)
    
    ax.set_title(f"{feat_cols[0]} vs {feat_cols[1]} {sample_info}")
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9)
    try:
        plt.tight_layout()
    except:
        pass  # tight_layout failed, matplotlib will handle it
    st.pyplot(fig)
    plt.close(fig)
    
    if len(anom) > 0:
        st.caption(f"🔴 {len(anom)} anomalies shown as red X marks | ✓ {len(norm):,} normal points ({sample_info})")


# ===== PCA =====
if len(feat_cols) >= 2 and len(display_df) > 1:
    st.subheader("PCA (Principal Component Analysis)")
    try:
        pca_data = display_df[feat_cols].astype(float)
        valid_rows = pca_data.dropna()
        if len(valid_rows) < 2:
            st.warning("Not enough valid data for PCA after removing NaN values")
        else:
            pca = PCA(n_components=2)
            X_pca = pca.fit_transform(valid_rows)
            
            pca_df = pd.DataFrame({
                "PC1": X_pca[:, 0], "PC2": X_pca[:, 1],
                num_col: display_df.loc[valid_rows.index, num_col].values,
                "Flag": display_df.loc[valid_rows.index, "AnomalyFlag"].values
            })
            
            norm_pca = pca_df[pca_df["Flag"] == "Normal"]
            anom_pca = pca_df[pca_df["Flag"] == "Anomaly"]
            
            # For large datasets, sample normal points
            MAX_NORMAL_POINTS = 1000
            if len(norm_pca) > MAX_NORMAL_POINTS:
                norm_pca_sampled = norm_pca.sample(n=MAX_NORMAL_POINTS, random_state=42)
                sample_info = f"(showing {MAX_NORMAL_POINTS:,} of {len(norm_pca):,} normal points)"
            else:
                norm_pca_sampled = norm_pca
                sample_info = ""
            
            fig, ax = plt.subplots(figsize=(14, 7))
            
            if len(norm_pca_sampled) > 0:
                sns.scatterplot(data=norm_pca_sampled, x="PC1", y="PC2", hue=num_col, ax=ax, 
                               palette="Set2", s=10, alpha=0.7, edgecolor="white", linewidth=0.5)
            if len(anom_pca) > 0:
                ax.scatter(anom_pca["PC1"], anom_pca["PC2"], color='red', s=10, marker='X', 
                          edgecolors='darkred', linewidth=2, alpha=0.95, label='ANOMALY', zorder=5)
            
            ax.set_title(f"PCA (2D Projection) {sample_info}")
            ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)")
            ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)")
            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9)
            try:
                plt.tight_layout()
            except:
                pass  # tight_layout failed, matplotlib will handle it
            st.pyplot(fig)
            plt.close(fig)
            
            total_variance = pca.explained_variance_ratio_[0] + pca.explained_variance_ratio_[1]
            if len(anom_pca) > 0:
                st.caption(f"🔴 {len(anom_pca)} anomalies | ✓ {len(norm_pca):,} normal ({sample_info}) | 📊 {total_variance:.0%} variance explained by PC1+PC2")
            else:
                st.caption(f"✓ {len(norm_pca):,} normal points ({sample_info}) | 📊 {total_variance:.0%} variance explained")
    except Exception as e:
        st.warning(f"Could not create PCA: {e}")

# ===== DISTRIBUTION =====
if len(display_df) > 1 and len(feat_cols) > 0:
    st.subheader(f"Distribution: {feat_cols[0]}")
    try:
        plot_data = display_df[[num_col, feat_cols[0]]].dropna()
        if len(plot_data) > 0:
            col1, col2 = st.columns([3, 1])
            
            with col2:
                plot_type = st.radio(
                    "Plot type:",
                    ["Box Plot", "Violin Plot", "Histogram"],
                    label_visibility="collapsed"
                )
            
            with col1:
                fig, ax = plt.subplots(figsize=(14, 5))
                
                if plot_type == "Box Plot":
                    sns.boxplot(data=plot_data, x=num_col, y=feat_cols[0], hue=num_col, ax=ax, palette="Set2", legend=False)
                    ax.set_title(f"Box Plot - {feat_cols[0]} by {num_col}")
                
                elif plot_type == "Violin Plot":
                    sns.violinplot(data=plot_data, x=num_col, y=feat_cols[0], hue=num_col, ax=ax, palette="Set2", legend=False)
                    ax.set_title(f"Violin Plot - {feat_cols[0]} by {num_col} (better for large datasets)")
                
                else:  # Histogram
                    for ptype in sorted(plot_data[num_col].unique()):
                        data = plot_data[plot_data[num_col] == ptype][feat_cols[0]]
                        ax.hist(data, alpha=0.5, label=ptype, bins=30)
                    ax.set_xlabel(feat_cols[0])
                    ax.set_ylabel("Count")
                    ax.set_title(f"Histogram - {feat_cols[0]} by {num_col}")
                    ax.legend()
                
                try:
                    plt.tight_layout()
                except:
                    pass  # tight_layout failed, matplotlib will handle it
                st.pyplot(fig)
                plt.close(fig)
                
                # Show summary statistics
                stats_data = []
                for ptype in sorted(plot_data[num_col].unique()):
                    subset = plot_data[plot_data[num_col] == ptype][feat_cols[0]]
                    stats_data.append({
                        "Type": ptype,
                        "Count": len(subset),
                        "Mean": subset.mean(),
                        "Std Dev": subset.std(),
                        "Min": subset.min(),
                        "Max": subset.max()
                    })
                
                st.caption("Summary Statistics:")
                st.dataframe(pd.DataFrame(stats_data), use_container_width=True)
        else:
            st.warning("No valid data for distribution plot")
    except Exception as e:
        st.warning(f"Could not create distribution: {e}")

# ===== HEATMAP =====
if len(feat_cols) >= 2 and len(display_df) > 1:
    st.subheader("Correlation")
    try:
        corr_data = display_df[feat_cols].astype(float)
        valid_corr = corr_data.dropna()
        
        if len(valid_corr) > 1:
            fig, ax = plt.subplots(figsize=(8, 6))
            corr = valid_corr.corr()
            sns.heatmap(corr, annot=True, cmap="coolwarm", center=0, square=True, ax=ax, fmt=".2f")
            try:
                plt.tight_layout()
            except:
                pass  # tight_layout failed, matplotlib will handle it
            st.pyplot(fig)
            plt.close(fig)
        else:
            st.warning("Not enough valid data for correlation matrix")
    except Exception as e:
        st.warning(f"Could not create heatmap: {e}")

# ===== SUMMARY =====
st.subheader("Summary")
c1, c2, c3 = st.columns(3)
c1.metric("Total", len(df_analysis))
c2.metric("Normal", len(df_analysis[df_analysis["AnomalyFlag"] == "Normal"]))
c3.metric("Anomalies", len(suspects))

if len(suspects) > 0:
    st.write("**By Part Type:**")
    try:
        summary = df_analysis.groupby(num_col)["AnomalyFlag"].value_counts().unstack(fill_value=0)
        st.dataframe(summary)
    except:
        pass
