
import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st

# Simulate dataset
np.random.seed(42)

def generate_measurements_and_params(part_type, n):
    if part_type == "PN-A":
        meas1 = np.random.normal(10.0, 0.05, n)
        meas2 = np.random.normal(6.0, 0.02, n)
        temp = np.random.normal(200, 2, n)
        pressure = np.random.normal(5.0, 0.2, n)
        speed = np.random.normal(100, 5, n)
    elif part_type == "PN-B":
        meas1 = np.random.normal(9.5, 0.03, n)
        meas2 = np.random.normal(6.5, 0.03, n)
        temp = np.random.normal(190, 2, n)
        pressure = np.random.normal(4.5, 0.2, n)
        speed = np.random.normal(95, 5, n)
    else:  # PN-C
        meas1 = np.random.normal(10.2, 0.04, n)
        meas2 = np.random.normal(5.8, 0.01, n)
        temp = np.random.normal(210, 2, n)
        pressure = np.random.normal(5.5, 0.2, n)
        speed = np.random.normal(105, 5, n)
    return meas1, meas2, temp, pressure, speed

records = []
for i in range(500):
    declared_label = np.random.choice(["PN-A", "PN-B", "PN-C"])
    meas1, meas2, temp, pressure, speed = generate_measurements_and_params(declared_label, 1)
    records.append({
        "PartID": f"P{i+1:03d}",
        "DeclaredPartNo": declared_label,
        "Meas1": round(meas1[0], 4),
        "Meas2": round(meas2[0], 4),
        "Temperature": round(temp[0], 2),
        "Pressure": round(pressure[0], 3),
        "Speed": round(speed[0], 1),
        "Machine": np.random.choice(["M1", "M2"]),
        "Tool": np.random.choice(["T1", "T2", "T3"]),
        "Operator": np.random.choice(["O123", "O124", "O125"])
    })

df = pd.DataFrame(records)

# Isolation Forest per DeclaredPartNo
df["AnomalyFlag"] = "Normal"
for part in df["DeclaredPartNo"].unique():
    subset = df[df["DeclaredPartNo"] == part]
    model = IsolationForest(contamination=0.1, random_state=42)
    features = subset[["Meas1", "Meas2", "Temperature", "Pressure", "Speed"]]
    preds = model.fit_predict(features)
    df.loc[subset.index, "AnomalyFlag"] = np.where(preds == -1, "Anomaly", "Normal")

# Save suspected mislabels
suspects = df[df["AnomalyFlag"] == "Anomaly"]
df.to_csv("all_data_with_anomalies.csv", index=False)
suspects.to_csv("suspected_mislabels.csv", index=False)

# Streamlit app
st.title("Mislabel Detection with Parameters")

show_anomalies_only = st.checkbox("Show only suspected mislabels", value=False)

display_df = suspects if show_anomalies_only else df
st.dataframe(display_df, use_container_width=True, height=500)

st.subheader("Scatter Plot: Meas1 vs Meas2")
sns.set(style="whitegrid")
fig1, ax1 = plt.subplots()
sns.scatterplot(data=display_df, x="Meas1", y="Meas2", hue="DeclaredPartNo", style="AnomalyFlag", ax=ax1)
st.pyplot(fig1)

st.subheader("PCA Visualization")
pca = PCA(n_components=2)
X = df[["Meas1", "Meas2", "Temperature", "Pressure", "Speed"]]
X_pca = pca.fit_transform(X)
df["PC1"], df["PC2"] = X_pca[:, 0], X_pca[:, 1]
fig2, ax2 = plt.subplots()
sns.scatterplot(data=df, x="PC1", y="PC2", hue="DeclaredPartNo", style="AnomalyFlag", ax=ax2)
st.pyplot(fig2)

st.subheader("Boxplot of Meas1 by Part Type")
fig3, ax3 = plt.subplots()
sns.boxplot(data=df, x="DeclaredPartNo", y="Meas1", ax=ax3)
st.pyplot(fig3)

st.subheader("Correlation Heatmap")
fig4, ax4 = plt.subplots()
sns.heatmap(df[["Meas1", "Meas2", "Temperature", "Pressure", "Speed"]].corr(), annot=True, cmap="coolwarm", ax=ax4)
st.pyplot(fig4)

st.success("Exported all data to 'all_data_with_anomalies.csv' and suspected mislabels to 'suspected_mislabels.csv'")
