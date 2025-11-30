import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.cluster import KMeans

# CONFIG
st.set_page_config(page_title="Voice Data Analysis", layout="wide")
DATA_PATH = "final_project_data/features.csv"


# LOAD DATA
@st.cache_data
def load_data():
    return pd.read_csv(DATA_PATH)


try:
    df = load_data()
except FileNotFoundError:
    st.error("Please run the 'prepare_data.py' script first!")
    st.stop()

# SIDEBAR NAVIGATION
st.sidebar.title("Project Navigation")
page = st.sidebar.radio("Go to", ["1. Overview", "2. Data Exploration", "3. Analysis (Clustering)", "4. Conclusions"])

# --- PAGE 1: OVERVIEW ---
if page == "1. Overview":
    st.title("🗣️ Project Overview")
    st.markdown("""
    **Dataset:** Japanese Eroge Voice Dataset  
    **Goal:** Analyze audio characteristics (loudness, duration, pitch) to cluster voice types.
    """)

    st.subheader("Dataset Sample")
    st.dataframe(df.head())

    st.subheader("Audio Preview")
    # Simple preview of first 3 items
    cols = st.columns(3)
    for i in range(3):
        row = df.iloc[i]
        with cols[i]:
            st.caption(f"ID: {row['id']}")
            st.audio(row['path'])
            st.text(row['text'])

# --- PAGE 2: EXPLORATION ---
elif page == "2. Data Exploration":
    st.title("📊 Data Exploration & Cleaning")

    st.markdown("### Feature Distribution")
    st.write("We extracted features using **Librosa** to quantify the audio.")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Distribution of Duration")
        fig = px.histogram(df, x="duration", nbins=30, title="How long are the clips?")
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("Distribution of Loudness (RMS)")
        fig = px.histogram(df, x="loudness", nbins=30, color_discrete_sequence=['orange'],
                           title="How loud are the clips?")
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("### Correlation Heatmap")
    st.write("Checking if longer clips are louder...")
    # Select only numeric columns for correlation
    numeric_df = df[["duration", "loudness", "roughness", "brightness"]]
    corr = numeric_df.corr()
    fig_corr = px.imshow(corr, text_auto=True, title="Feature Correlation")
    st.plotly_chart(fig_corr)

# --- PAGE 3: ANALYSIS (THE BIG GRADE) ---
elif page == "3. Analysis (Clustering)":
    st.title("🧮 K-Means Clustering Analysis")
    st.markdown("We use **K-Means Clustering** to group voice clips based on their audio features.")

    # 1. Select Features
    features = ['duration', 'loudness', 'roughness', 'brightness']
    X = df[features]

    # 2. Interactive K-Selection
    k = st.sidebar.slider("Number of Clusters (K)", 2, 6, 3)

    # 3. Run Algorithm
    kmeans = KMeans(n_clusters=k, random_state=42)
    df['cluster'] = kmeans.fit_predict(X)

    # 4. Visualize Results
    st.subheader(f"Cluster Visualization (K={k})")

    # User chooses axes
    x_axis = st.selectbox("X Axis", features, index=0)
    y_axis = st.selectbox("Y Axis", features, index=1)

    fig = px.scatter(
        df, x=x_axis, y=y_axis,
        color=df['cluster'].astype(str),
        hover_data=['text', 'id'],
        title=f"Clustering based on {x_axis} vs {y_axis}"
    )
    st.plotly_chart(fig, use_container_width=True)

    # 5. Cluster Interpretation (Interactive)
    st.subheader("Inspect a Cluster")
    selected_cluster = st.selectbox("Select Cluster to Listen", sorted(df['cluster'].unique()))

    cluster_data = df[df['cluster'] == selected_cluster]
    st.write(f"This cluster contains {len(cluster_data)} audio files.")

    # Play samples from this cluster
    st.write("### Audio Samples from this Cluster")
    cols = st.columns(3)
    for i, (_, row) in enumerate(cluster_data.head(3).iterrows()):
        with cols[i]:
            st.audio(row['path'])
            st.caption(f"Loudness: {row['loudness']:.4f}")

# --- PAGE 4: CONCLUSIONS ---
elif page == "4. Conclusions":
    st.title("📝 Conclusions")
    st.info("Based on our K-Means analysis, we found distinct patterns in the voice data.")

    st.markdown("""
    * **Cluster 0 (Example):** Tend to be short, loud exclamations.
    * **Cluster 1 (Example):** Longer, softer spoken sentences.
    * **Recommendation:** For future TTS models, we should filter out Cluster 0 to ensure high-quality, stable training data.
    """)