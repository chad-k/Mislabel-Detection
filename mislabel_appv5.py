import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st

st.set_page_config(page_title="Mislabel Detection with Smart Contamination", layout="wide")

st.title("Manufacturing Mislabel Detection (Smart Contamination)")


# ===== DYNAMIC CONTAMINATION DETECTION =====
def estimate_optimal_contamination_per_type(anomaly_scores_dict: dict) -> tuple[dict, str]:
    """
    Intelligently estimate optimal contamination rates per part type.
    
    Args:
        anomaly_scores_dict: {part_type: anomaly_scores_array}
    
    Returns:
        ({part_type: contamination_rate}, explanation)
    """
    contamination_map = {}
    explanations = []
    
    for part_type, scores in anomaly_scores_dict.items():
        scores_clean = np.array(scores)[np.isfinite(scores)]
        
        if len(scores_clean) < 5:
            # Too few samples - use default
            contamination_map[part_type] = 0.15
            explanations.append(f"{part_type}: <5 samples → 15% (default)")
            continue
        
        # Method 1: ELBOW (gap detection)
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
        
        # Method 2: Z-score (statistical)
        mean_score = np.mean(scores_clean)
        std_score = np.std(scores_clean)
        
        threshold_z = mean_score + 1.5 * std_score
        n_anomalies = np.sum(scores_clean > threshold_z)
        contamination_z = max(0.05, min(0.25, n_anomalies / len(scores_clean)))
        
        if 0.05 <= contamination_z <= 0.25:
            contamination_map[part_type] = contamination_z
            explanations.append(f"{part_type}: Z-score {contamination_z*100:.1f}%")
            continue
        
        # Method 3: IQR (robust)
        q1, q3 = np.percentile(scores_clean, [25, 75])
        iqr = q3 - q1
        outliers = (scores_clean < q1 - 1.5 * iqr) | (scores_clean > q3 + 1.5 * iqr)
        n_outliers = np.sum(outliers)
        contamination_iqr = max(0.05, min(0.25, n_outliers / len(scores_clean)))
        
        if 0.05 <= contamination_iqr <= 0.25:
            contamination_map[part_type] = contamination_iqr
            explanations.append(f"{part_type}: IQR {contamination_iqr*100:.1f}%")
            continue
        
        # Method 4: Fallback
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
    # Suggest good columns (>90% complete)
    good_cols = completeness_df[completeness_df["Complete %"] > 90]["Column"].tolist()
    
    st.write(f"**Recommended features ({len(good_cols)}):** Columns with >90% data")
    
    feat_cols = st.multiselect(
        "Select 3+ Features (choose columns with high completeness)",
        cols,
        default=good_cols[:min(5, len(good_cols))],
        help="These numeric features will be analyzed for anomalies within each part type."
    )
    
    # Warn if selected features have low completeness
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
    with st.expander("🔍 Diagnostic Info", expanded=True):
        st.write("**Before conversion:**")
        diag_before = pd.DataFrame({
            "Feature": feat_cols,
            "Type": [str(df_analysis[col].dtype) for col in feat_cols],
            "Non-Null": [df_analysis[col].notna().sum() for col in feat_cols],
            "Sample": [str(df_analysis[col].iloc[0])[:50] for col in feat_cols]
        })
        st.dataframe(diag_before, use_container_width=True)
    
    # Convert feature columns to numeric, coercing errors
    for col in feat_cols:
        df_analysis[col] = pd.to_numeric(df_analysis[col], errors='coerce')
    
    with st.expander("🔍 After Numeric Conversion", expanded=True):
        st.write("**After pd.to_numeric():**")
        diag_after = pd.DataFrame({
            "Feature": feat_cols,
            "Type": [str(df_analysis[col].dtype) for col in feat_cols],
            "Non-Null": [df_analysis[col].notna().sum() for col in feat_cols],
            "NaN Count": [df_analysis[col].isna().sum() for col in feat_cols],
            "Sample": [str(df_analysis[col].iloc[0])[:50] for col in feat_cols]
        })
        st.dataframe(diag_after, use_container_width=True)
    
    # Check for ANY missing values
    missing_per_col = {col: df_analysis[col].isna().sum() for col in feat_cols}
    total_missing = sum(missing_per_col.values())
    
    st.write(f"**Total missing values across all features:** {total_missing}")
    
    # Drop rows with missing values in features
    rows_before = len(df_analysis)
    df_analysis = df_analysis.dropna(subset=feat_cols)
    rows_after = len(df_analysis)
    rows_dropped = rows_before - rows_after
    
    if rows_dropped > 0:
        st.warning(f"⚠️ Removed {rows_dropped} rows with missing values in features ({(rows_dropped/rows_before)*100:.1f}%)")
    else:
        st.success("✓ No rows removed - all features have valid data!")
    
    if len(df_analysis) == 0:
        st.error("❌ No valid data after removing missing values.")
        st.error("**Why this happened:**")
        st.write("- The selected columns may contain values that couldn't be converted to numbers (e.g., text, symbols)")
        st.write("- There may be hidden missing values (spaces, empty strings, etc.)")
        st.write("- Try deselecting some columns and clicking 'Analyze Data' again")
        st.stop()
    
    st.info(f"✓ Working with {len(df_analysis)} rows")
    
    # ===== DYNAMIC CONTAMINATION DETECTION =====
    st.subheader("🤖 Smart Contamination Detection")
    
    # First pass: calculate anomaly scores for each part type to estimate contamination
    anomaly_scores_by_type = {}
    
    for part in df_analysis[num_col].unique():
        idx = df_analysis[df_analysis[num_col] == part].index
        subset = df_analysis.loc[idx, feat_cols].astype(float)
        
        if len(subset) > 1:
            # Use initial model to get scores (will refit later with optimal contamination)
            model_initial = IsolationForest(contamination=0.1, random_state=42)
            scores = -model_initial.fit_predict(subset)  # Convert predictions to continuous scores
            decision_scores = model_initial.decision_function(subset)
            anomaly_scores_by_type[part] = -decision_scores
    
    # Estimate optimal contamination rates
    contamination_map, contamination_explanation = estimate_optimal_contamination_per_type(anomaly_scores_by_type)
    
    st.session_state.contamination_map = contamination_map
    
    # Display the decision
    st.info(
        f"**Automatically Detected Contamination Rates:**\n\n"
        f"{contamination_explanation}"
    )
    
    # Allow override
    override = st.checkbox(
        "Override contamination rates?",
        value=False,
        help="Enable to manually set contamination rates per part type"
    )
    
    if override:
        st.write("Set contamination rate for each part type (2-30%):")
        for part in contamination_map:
            contamination_map[part] = st.slider(
                f"{part}",
                min_value=0.02,
                max_value=0.30,
                value=contamination_map[part],
                step=0.01,
                format="%.2f"
            )
    
    # ===== ANOMALY DETECTION (SECOND PASS WITH OPTIMAL CONTAMINATION) =====
    st.subheader("Running Anomaly Detection...")
    
    df_analysis["AnomalyFlag"] = "Normal"
    df_analysis["AnomalyScore"] = 0.0
    
    for part in df_analysis[num_col].unique():
        idx = df_analysis[df_analysis[num_col] == part].index
        subset = df_analysis.loc[idx, feat_cols].astype(float)
        
        if len(subset) < 2:
            # BUG FIX: Handle single-sample part types explicitly
            st.warning(f"⚠️ Part type '{part}' has only {len(subset)} sample(s) - cannot detect anomalies")
            continue
        
        # Get optimal contamination for this part type
        cont_rate = contamination_map.get(part, 0.10)
        
        # Fit model with optimal contamination
        model = IsolationForest(contamination=cont_rate, random_state=42)
        preds = model.fit_predict(subset)
        decision_scores = model.decision_function(subset)
        
        df_analysis.loc[idx, "AnomalyFlag"] = ["Anomaly" if p == -1 else "Normal" for p in preds]
        df_analysis.loc[idx, "AnomalyScore"] = -decision_scores  # Higher = more anomalous
    
    st.session_state.df_analysis = df_analysis
    st.session_state.id_col = id_col
    st.success(f"✓ Analysis complete ({len(df_analysis)} rows)")
else:
    df_analysis = st.session_state.df_analysis
    id_col = st.session_state.id_col
    num_col = st.session_state.num_col
    feat_cols = st.session_state.feat_cols
    contamination_map = st.session_state.contamination_map

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
            
            # Add threshold line
            threshold = np.percentile(part_data["AnomalyScore"], (1 - contamination_map[part_type]) * 100)
            ax.axvline(threshold, color='red', linestyle='--', linewidth=2, label=f'Threshold ({contamination_map[part_type]*100:.0f}%)')
            
            ax.set_title(f"{part_type} (n={len(part_data)})")
            ax.set_xlabel("Anomaly Score")
            ax.set_ylabel("Count")
            ax.legend()
    
    plt.tight_layout()
    st.pyplot(fig)

# ===== EXPLAIN ROW =====
st.subheader("Explain a Row")
part_ids = display_df[id_col].unique().tolist() if len(display_df) > 0 else []
if part_ids:
    selected = st.selectbox("Select Part ID", part_ids, help="Pick a part to see why it was flagged (or not)")
    try:
        row = df_analysis[df_analysis[id_col] == selected].iloc[0]
        
        st.write(f"**ID:** {selected} | **Type:** {row[num_col]} | **Flag:** {row['AnomalyFlag']} | **Score:** {row['AnomalyScore']:.3f}")
        
        base = df_analysis[df_analysis[num_col] == row[num_col]]
        means = base[feat_cols].mean()
        stds = base[feat_cols].std(ddof=1)  # BUG FIX: Use ddof=1 for sample std
        
        vals = pd.to_numeric(row[feat_cols], errors='coerce')
        
        # BUG FIX: Handle z-score calculation more robustly
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
        
        st.dataframe(exp_df.style.format({c: "{:.3f}" for c in exp_df.columns[1:]}), use_container_width=True)
        
        fig, ax = plt.subplots(figsize=(10, 4))
        valid = exp_df[np.isfinite(exp_df["|Z|"])]
        if len(valid) > 0:
            sns.barplot(data=valid, x="Feature", y="|Z|", ax=ax, palette="husl")
            ax.set_ylabel("|Z-score|")
            ax.set_title(f"Feature Impact: {selected}")
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            st.pyplot(fig)
        else:
            st.warning("Could not calculate Z-scores for this part")
    except Exception as e:
        st.warning(f"Error: {e}")

# ===== SCATTER PLOT =====
if len(feat_cols) >= 2 and len(display_df) > 0:
    st.subheader("Scatter Plot")
    # BUG FIX: Handle NaN values in scatter plot
    norm = display_df[display_df["AnomalyFlag"] == "Normal"].dropna(subset=[feat_cols[0], feat_cols[1]])
    anom = display_df[display_df["AnomalyFlag"] == "Anomaly"].dropna(subset=[feat_cols[0], feat_cols[1]])
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    if len(norm) > 0:
        sns.scatterplot(data=norm, x=feat_cols[0], y=feat_cols[1], hue=num_col, 
                      ax=ax, palette="Set2", s=80, alpha=0.6)
    if len(anom) > 0:
        ax.scatter(anom[feat_cols[0]], anom[feat_cols[1]], color='red', s=250, marker='X', 
                  edgecolors='darkred', linewidth=2, alpha=0.95, label='ANOMALY', zorder=5)
    
    ax.set_title(f"{feat_cols[0]} vs {feat_cols[1]}")
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    st.pyplot(fig)

# ===== PCA =====
if len(feat_cols) >= 2 and len(display_df) > 1:
    st.subheader("PCA")
    try:
        # BUG FIX: Better handling of missing values in PCA
        pca_data = display_df[feat_cols].astype(float)
        
        # Check if we have enough valid data
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
            
            fig, ax = plt.subplots(figsize=(12, 6))
            norm_pca = pca_df[pca_df["Flag"] == "Normal"]
            anom_pca = pca_df[pca_df["Flag"] == "Anomaly"]
            
            if len(norm_pca) > 0:
                sns.scatterplot(data=norm_pca, x="PC1", y="PC2", hue=num_col, ax=ax, palette="Set2", s=80, alpha=0.6)
            if len(anom_pca) > 0:
                ax.scatter(anom_pca["PC1"], anom_pca["PC2"], color='red', s=250, marker='X', 
                          edgecolors='darkred', linewidth=2, alpha=0.95, label='ANOMALY', zorder=5)
            
            ax.set_title("PCA")
            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            plt.tight_layout()
            st.pyplot(fig)
            st.caption(f"PC1: {pca.explained_variance_ratio_[0]:.1%} | PC2: {pca.explained_variance_ratio_[1]:.1%}")
    except Exception as e:
        st.warning(f"Could not create PCA: {e}")

# ===== BOXPLOT =====
if len(display_df) > 1 and len(feat_cols) > 0:
    st.subheader(f"Distribution: {feat_cols[0]}")
    try:
        # BUG FIX: Handle NaN in boxplot
        plot_data = display_df[[num_col, feat_cols[0]]].dropna()
        if len(plot_data) > 0:
            fig, ax = plt.subplots(figsize=(10, 5))
            sns.boxplot(data=plot_data, x=num_col, y=feat_cols[0], ax=ax, palette="Set2")
            plt.tight_layout()
            st.pyplot(fig)
        else:
            st.warning("No valid data for boxplot after removing NaN values")
    except Exception as e:
        st.warning(f"Could not create boxplot: {e}")

# ===== HEATMAP =====
if len(feat_cols) >= 2 and len(display_df) > 1:
    st.subheader("Correlation")
    try:
        # BUG FIX: Check for sufficient valid data
        corr_data = display_df[feat_cols].astype(float)
        valid_corr = corr_data.dropna()
        
        if len(valid_corr) > 1:
            fig, ax = plt.subplots(figsize=(8, 6))
            corr = valid_corr.corr()
            sns.heatmap(corr, annot=True, cmap="coolwarm", center=0, square=True, ax=ax, fmt=".2f")
            plt.tight_layout()
            st.pyplot(fig)
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
