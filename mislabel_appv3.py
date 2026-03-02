import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st

st.set_page_config(page_title="Mislabel Detection", layout="wide")

st.title("Manufacturing Mislabel Detection")

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

# ===== DATA LOADING =====
st.sidebar.header("Data Source")
data_source = st.sidebar.radio("Choose:", ["Demo Data", "Upload CSV"])

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
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

col1, col2 = st.columns(2)

with col1:
    id_opts = ["(Auto-Generate ID)"] + cols
    id_col_input = st.selectbox("Part ID Column", id_opts)
    id_col = None if id_col_input == "(Auto-Generate ID)" else id_col_input
    
    num_col = st.selectbox("Part Number/Label Column", cols)

with col2:
    feat_cols = st.multiselect(
        "Select 3+ Numeric Features",
        numeric_cols,
        default=numeric_cols[:min(5, len(numeric_cols))]
    )

# ===== ANALYSIS BUTTON =====
st.header("Step 2: Run Analysis")

analysis_ready = st.button("▶ Analyze Data", type="primary", use_container_width=True)

if analysis_ready:
    # Validate
    if len(feat_cols) < 3:
        st.error("Select at least 3 numeric features")
        st.stop()
    if not num_col:
        st.error("Select a Part Number column")
        st.stop()
    
    # Store in session state
    st.session_state.analysis_complete = True
    st.session_state.id_col = id_col
    st.session_state.num_col = num_col
    st.session_state.feat_cols = feat_cols
    st.session_state.df_analysis = None  # Reset

# Check if analysis has been run
if not st.session_state.get("analysis_complete", False):
    st.stop()

# ===== DO ANALYSIS (only once) =====
if st.session_state.df_analysis is None:
    id_col = st.session_state.id_col
    num_col = st.session_state.num_col
    feat_cols = st.session_state.feat_cols
    
    df_analysis = df.copy()
    
    # Generate ID if needed
    if id_col is None:
        df_analysis.insert(0, "_PartID", [f"Row_{i+1}" for i in range(len(df_analysis))])
        id_col = "_PartID"
    
    # Handle missing values
    missing = df_analysis[feat_cols].isnull().sum().sum()
    if missing > 0:
        st.warning(f"Removing {int(missing)} rows with missing values")
        df_analysis = df_analysis.dropna(subset=feat_cols)
    
    # Anomaly detection
    df_analysis["AnomalyFlag"] = "Normal"
    for part in df_analysis[num_col].unique():
        idx = df_analysis[df_analysis[num_col] == part].index
        subset = df_analysis.loc[idx, feat_cols].astype(float)
        if len(subset) > 1:
            model = IsolationForest(contamination=0.1, random_state=42)
            preds = model.fit_predict(subset)
            df_analysis.loc[idx, "AnomalyFlag"] = ["Anomaly" if p == -1 else "Normal" for p in preds]
    
    st.session_state.df_analysis = df_analysis
    st.session_state.id_col = id_col
    st.success(f"✓ Analysis complete ({len(df_analysis)} rows)")
else:
    df_analysis = st.session_state.df_analysis
    id_col = st.session_state.id_col
    num_col = st.session_state.num_col
    feat_cols = st.session_state.feat_cols

# ===== EXPORT =====
st.sidebar.header("Export")
suspects = df_analysis[df_analysis["AnomalyFlag"] == "Anomaly"]

col_e1, col_e2 = st.sidebar.columns(2)
with col_e1:
    st.download_button("📥 All Data", df_analysis.to_csv(index=False), "data.csv", "text/csv")
with col_e2:
    st.download_button("📥 Anomalies", suspects.to_csv(index=False), "anomalies.csv", "text/csv")

if st.sidebar.button("🔄 Reset", use_container_width=True):
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

# ===== EXPLAIN ROW =====
st.subheader("Explain a Row")
part_ids = display_df[id_col].unique().tolist() if len(display_df) > 0 else []
if part_ids:
    selected = st.selectbox("Select Part ID", part_ids)
    try:
        row = df_analysis[df_analysis[id_col] == selected].iloc[0]
        
        st.write(f"**ID:** {selected} | **Type:** {row[num_col]} | **Flag:** {row['AnomalyFlag']}")
        
        base = df_analysis[df_analysis[num_col] == row[num_col]]
        means = base[feat_cols].mean()
        stds = base[feat_cols].std(ddof=0).replace(0, np.nan)
        
        vals = pd.to_numeric(row[feat_cols], errors='coerce')
        z = (vals - means) / stds
        
        exp_df = pd.DataFrame({
            "Feature": feat_cols,
            "Value": vals.values,
            "Mean": means.values,
            "Std": stds.values,
            "|Z|": np.abs(z).values
        }).sort_values("|Z|", ascending=False)
        
        st.dataframe(exp_df.style.format({c: "{:.3f}" for c in exp_df.columns[1:]}), use_container_width=True)
        
        # Bar chart
        fig, ax = plt.subplots(figsize=(10, 4))
        valid = exp_df[exp_df["|Z|"].notna()]
        if len(valid) > 0:
            sns.barplot(data=valid, x="Feature", y="|Z|", ax=ax, palette="husl")
            ax.set_ylabel("|Z-score|")
            ax.set_title(f"Feature Impact: {selected}")
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            st.pyplot(fig)
    except Exception as e:
        st.warning(f"Error: {e}")

# ===== SCATTER PLOT =====
if len(feat_cols) >= 2 and len(display_df) > 0:
    st.subheader("Scatter Plot")
    norm = display_df[display_df["AnomalyFlag"] == "Normal"]
    anom = display_df[display_df["AnomalyFlag"] == "Anomaly"]
    
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
        pca = PCA(n_components=2)
        X = display_df[feat_cols].astype(float).fillna(display_df[feat_cols].mean())
        X_pca = pca.fit_transform(X)
        
        pca_df = pd.DataFrame({
            "PC1": X_pca[:, 0], "PC2": X_pca[:, 1],
            num_col: display_df[num_col].values, "Flag": display_df["AnomalyFlag"].values
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
if len(display_df) > 1:
    st.subheader(f"Distribution: {feat_cols[0]}")
    try:
        fig, ax = plt.subplots(figsize=(10, 5))
        sns.boxplot(data=display_df, x=num_col, y=feat_cols[0], ax=ax, palette="Set2")
        plt.tight_layout()
        st.pyplot(fig)
    except Exception as e:
        st.warning(f"Could not create boxplot: {e}")

# ===== HEATMAP =====
if len(feat_cols) >= 2 and len(display_df) > 1:
    st.subheader("Correlation")
    try:
        fig, ax = plt.subplots(figsize=(8, 6))
        corr = display_df[feat_cols].astype(float).corr()
        sns.heatmap(corr, annot=True, cmap="coolwarm", center=0, square=True, ax=ax, fmt=".2f")
        plt.tight_layout()
        st.pyplot(fig)
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