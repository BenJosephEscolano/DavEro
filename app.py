import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA

# --- CONFIGURATION ---
st.set_page_config(page_title="Voice Style Analysis", layout="wide")
DATA_PATH = "final_project_data/multimodal_features.csv"


# --- LOADER ---
@st.cache_data
def load_data():
    try:
        df = pd.read_csv(DATA_PATH)
        return df
    except FileNotFoundError:
        return None


df_raw = load_data()

# --- ERROR HANDLING ---
if df_raw is None:
    st.error("🚨 Data not found! Please run 'prepare_data.py' first.")
    st.stop()

# --- SIDEBAR ---
with st.sidebar:
    st.title("Final Project")
    st.caption("CS365-F2 | Group: Baithon") # This creates the faded look

page = st.sidebar.radio("Navigate", [
    "1. Project Overview",
    "2. Data Exploration & Methodology",
    "3. Analysis Part 1: The Style Tokens (Clustering)",
    "4. Analysis Part 2: Engineering Intimacy (Physics)",
    "5. Analysis Part 3: Cross-Modal Alignment (Text)"
])

# --- GLOBAL DATA PREP (Normalize Once) ---
# We normalize features 0-1 for all analysis to keep math consistent
features = ['loudness', 'breathiness', 'pitch', 'brightness', 'tonality']
scaler = MinMaxScaler()
df_norm = df_raw.copy()
df_norm[features] = scaler.fit_transform(df_raw[features])

# --- PAGE 1: OVERVIEW ---
if page == "1. Project Overview":
    st.title("Engineering Intimacy: The Paralinguistics of Voice")
    st.markdown("### What it's about: Project Overview")
    # 1. THE HOOK (The Hybrid Approach)
    with st.expander("Abstract: Research Question & Hypothesis"):
        st.info("""
        **Research Question:** Can we mathematically "Engineer" the feeling of intimacy?
    
        **The Hypothesis:** Intimacy is not random. It is a specific **Paralinguistic Code** composed of breathiness, proximity, and rhythm. 
        By analyzing **2,000** clips of hyper-stylized voice acting, we aim to reverse-engineer this code to prove that "Emotion" is just a vector of numbers.
        """)

    # 2. THE TRANSITION (Why this dataset?)
    st.markdown("""
    **Why this dataset?** We selected the *NandemoGHS* dataset because it represents **"Performative Intimacy."** The voice actors are not just speaking; they are utilizing extreme paralinguistic control to simulate closeness. This makes it the perfect laboratory to study the **Proximity Effect**.
    """)

    st.markdown("### Why its relevant: A Digital Prosthetic for Loneliness")
    st.markdown("""
        > "Against the backdrop of a global **male loneliness epidemic**, the demand for 'synthetic intimacy'—from ASMR to AI companions—has skyrocketed. 
        > This dataset represents the industrial response to that crisis: a highly optimized, paralinguistic simulation of closeness designed to soothe social isolation. 
        > By decoding the acoustic mechanics of this performance, we are not just analyzing voice acting; **we are reverse-engineering the modern digital prosthetic for human connection.**"
        """)

    st.divider()

    # 3. THE TOOLKIT (Linguistics meets Physics)
    st.markdown("### Acoustic Features")
    st.write("We deconstruct the voice into 5 acoustic dimensions, mapping physics to linguistic meaning:")

    # This table bridges your two interests perfectly
    data_dict = pd.DataFrame([
        {"Feature": "Breathiness (ZCR)", "Linguistic Concept": "Phonation Type",
         "Role": "The 'Whisper Factor'. Essential for simulating physical closeness."},
        {"Feature": "Loudness (RMS)", "Linguistic Concept": "Prosodic Stress",
         "Role": "Inverse of Intimacy. Low energy creates a 'Private Space'."},
        {"Feature": "Pitch (F0)", "Linguistic Concept": "Intonation",
         "Role": "Defines the archetype (e.g., High=Vulnerable, Low=Dominant)."},
        {"Feature": "Speed (Char/Sec)", "Linguistic Concept": "Tempo",
         "Role": "Information density. Slow tempo signals comfort/seduction."},
        {"Feature": "Tonality (Flatness)", "Linguistic Concept": "Vocal Stability",
         "Role": "Tremble/Jitter signals emotional instability (High Arousal)."}
    ])
    st.table(data_dict)

    st.divider()

    # 4. DATASET METRICS
    col1, col2, col3 = st.columns(3)
    col1.metric("Sample Size", len(df_raw), "Clips Analyzed")
    col2.metric("Avg Prosodic Pitch", f"{df_raw['pitch'].mean():.0f} Hz")
    col3.metric("Semantic Density", "Sparse",
                help="Most files contain functional dialogue, with rare spikes of high emotion.")

    # 5. RAW DATA INSPECTION
    with st.expander("🔍 Inspect the Raw Signal"):
        st.dataframe(df_raw.head(50))

# ... inside app.py ...

elif page == "2. Data Exploration & Methodology":
    st.title("Data Exploration & Methodology")
    st.markdown("""
    To analyze "Intimacy," we needed to transform raw audio into structured data. 
    This section details our **ETL (Extract, Transform, Load) Pipeline** and explores the statistical distribution of the resulting features.
    """)

    # --- SECTION 1: METHODOLOGY PIPELINE ---
    st.subheader("1. The Engineering Pipeline")
    st.info("We utilized a 4-stage process to prepare the data for analysis:")

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown("#### 1. Ingestion")
        st.caption("Streamed from Hugging Face (`NandemoGHS`).")
        st.markdown("⬇**Shuffle Buffer:** 1k items to prevent bias.")
    with col2:
        st.markdown("#### 2. Pre-Processing")
        st.caption("Standardization for ASR.")
        st.markdown("**Resampling:** Downsampled to **16kHz Mono**.")
    with col3:
        st.markdown("#### 3. Extraction")
        st.caption("Spectral Analysis via `Librosa`.")
        st.markdown("**5 Features:** Pitch, RMS, ZCR, Centroid, Flatness.")
    with col4:
        st.markdown("#### 4. Cleaning")
        st.caption("Quality Control.")
        st.markdown("**Filter:** Removed silent clips ($F_0=0$) and null text.")

    st.divider()

    # --- SECTION 2: DATA INTEGRITY CHECK ---
    st.subheader("2. Data Integrity Check")

    c1, c2 = st.columns(2)
    with c1:
        st.write("**Missing Value Inspection:**")
        # Check for nulls (should be 0 if our script worked)
        nulls = df_raw.isnull().sum()
        if nulls.sum() == 0:
            st.success("✅ No missing values detected.")
        else:
            st.warning("⚠️ Missing values found (handled in visualizations).")
            st.dataframe(nulls)

    with c2:
        st.write("**Outlier Removal (Silence):**")
        st.caption("We filtered out clips where the Fundamental Frequency ($F_0$) was 0Hz (Silence/Noise).")
        st.metric("Cleaned Sample Size", len(df_raw))

    st.divider()

    # --- SECTION 3: FEATURE DISTRIBUTIONS ---
    st.subheader("3. Feature Distributions (Histograms)")
    st.markdown("How is the 'Style' distributed across the dataset?")

    # Interactive Selector
    feature_to_plot = st.selectbox(
        "Select Feature to Visualize",
        ["semantic_score", "pitch", "loudness", "breathiness", "speed"],
        format_func=lambda x: x.replace("_", " ").title()
    )

    # Logic for specific comments based on selection
    comment = ""
    if feature_to_plot == "semantic_score":
        comment = "👉 **Insight:** The distribution is **Right-Skewed** (Power Law). High intimacy is rare; most dialogue is functional."
    elif feature_to_plot == "pitch":
        comment = "👉 **Insight:** The distribution is **Bimodal** (Two peaks), suggesting distinct character types (High-pitched 'Moe' vs. Low-pitched 'Mature')."
    elif feature_to_plot == "breathiness":
        comment = "👉 **Insight:** High breathiness (>0.2) represents the 'ASMR/Whisper' zone."

    fig_hist = px.histogram(
        df_raw,
        x=feature_to_plot,
        nbins=30,
        title=f"Distribution of {feature_to_plot.replace('_', ' ').title()}",
        color_discrete_sequence=['#636EFA'],
        marginal="box"  # Adds a boxplot on top to show outliers
    )
    st.plotly_chart(fig_hist, use_container_width=True)
    st.info(comment)

    # --- SECTION 4: CORRELATION HEATMAP ---
    st.subheader("4. Correlation Heatmap")
    st.markdown("Do louder voices tend to have higher pitch? (Testing the Physics)")

    # Numeric columns only
    corr_cols = ['loudness', 'breathiness', 'pitch', 'brightness', 'tonality', 'speed', 'semantic_score']
    corr_matrix = df_raw[corr_cols].corr()

    fig_corr = px.imshow(
        corr_matrix,
        text_auto=".2f",
        color_continuous_scale="RdBu_r",  # Red = Positive, Blue = Negative
        aspect="auto",
        title="Feature Correlation Matrix"
    )
    st.plotly_chart(fig_corr, use_container_width=True)

    with st.expander("Show Interpretation of Heatmap"):
        st.markdown("""
        * **Breathiness & Brightness (+0.72):** The strongest link in the dataset. Whispering generates high-frequency "hiss," proving our texture analysis is accurate.
        * **Breathiness & Tonality (-0.47):** Strong negative correlation. As expected, "breathy" whispers lack tonal stability.
        * **Loudness & Pitch (-0.23):** Surprisingly weak negative correlation. This suggests variety: actors perform both "Low/Loud" (Dominant) and "High/Loud" (Panic) scenes.
        * **Semantic Score (~0.0):** Almost no correlation with acoustics. This proves that **what** is said is independent of **how** it is said—a sign of complex acting.
        """)
    st.divider()

    # --- SECTION 5: DETAILED METHODOLOGY (THE THEORY) ---
    st.subheader("5. Algorithm Methodology")
    st.markdown(
        "We employed a multi-modal approach, combining signal processing physics with linguistic rule-based analysis.")

    # A. Acoustic Extraction
    with st.expander("Acoustic Feature Extraction (Librosa)"):
        st.markdown("""
            **Objective:** To quantify the "Paralinguistic Signal."

            We treated the audio as a time-series signal and extracted features using the **Librosa** library. 
            Instead of using raw waveforms (which are high-dimensional and noisy), we extracted 5 spectral proxies:

            1.  **Energy (RMS):** Calculated as the root mean square of the waveform amplitude. Used as a proxy for **Arousal**.
            2.  **Texture (Zero-Crossing Rate):** The rate at which the signal changes sign. High rates indicate "noisy" signals (fricatives/whispers), used as a proxy for **Intimacy/Proximity**.
            3.  **Pitch ($F_0$):** Extracted using the **YIN Algorithm**. We filtered values below 60Hz and above 1000Hz to isolate human vocal range.
            4.  **Timbre (Spectral Centroid):** The "center of mass" of the spectrum. Used to distinguish "Bright/Anime" voices from "Dark/Mature" voices.
            5.  **Stability (Spectral Flatness):** Measures how noise-like a sound is. Low flatness indicates a pure tone; high flatness indicates breath or trembling.
            """)

    # B. Semantic Analysis
    with st.expander("Semantic Analysis (Rule-Based NLP)"):
        st.markdown("""
            **Objective:** To quantify the "Semantic Symbol" (Intent).

            Standard NLP models (BERT/Transformers) fail on this dataset due to the high frequency of non-grammatical utterances (stuttering, moaning, slang).

            **Our Solution: Domain-Specific Lexicon Analysis**
            We engineered a weighted dictionary of genre-specific triggers (e.g., *Suki, Aishiteru, Motto*). 
            To prevent double-counting (e.g., finding "Love" inside "Cute"), we implemented a **Recursive Longest-Match Consumption Algorithm**:
            1.  Sort lexicon phrases by length (Longest → Shortest).
            2.  Scan text for matches.
            3.  Score and **remove** the matched phrase from the string.
            4.  Repeat until the string is empty.
            """)

    # C. Unsupervised Learning
    with st.expander("Unsupervised Learning (K-Means & PCA)"):
        st.markdown("""
            **Objective:** To discover "Style Tokens" without human bias.

            We hypothesize that emotional archetypes are mathematically distinct clusters in high-dimensional feature space.
            * **Algorithm:** K-Means Clustering ($K=5$).
            * **Normalization:** Min-Max Scaling (0-1) to ensure Pitch (Hz) doesn't overpower Loudness (dB).
            * **Visualization:** We use **PCA (Principal Component Analysis)** to reduce the 5 dimensions down to 2D for visualization, proving that the clusters are distinct "islands" of emotion.
            """)
    # D. Feature Engineering (Physics)
    with st.expander("Feature Engineering (The Physics of Intimacy)"):
        st.markdown("""
        **Objective:** To Operationalize the "Proximity Effect."

        We hypothesized that "Intimacy" and "Intensity" are orthogonal vectors, similar to the **Russell Circumplex Model of Affect**.
        We engineered two composite metrics based on audio physics:

        1.  **Calculated Intimacy (Proximity):** * *Formula:* `(Breathiness * 1.5) + ((1 - Loudness) * 1.0)`
            * *Logic:* Physical closeness creates the "Proximity Effect" (Bass boost + Airiness). Therefore, High Breathiness + Low Volume = High Intimacy.

        2.  **Calculated Intensity (Arousal):**
            * *Formula:* `(Loudness * 1.5) + (Pitch * 1.0)`
            * *Logic:* High physiological arousal results in increased sub-glottal pressure (Loudness) and vocal cord tension (Pitch).
        """)

    # E. Cross-Modal Validation
    with st.expander("Cross-Modal Validation (Signal vs. Symbol)"):
        st.markdown("""
        **Objective:** To measure Acting Consistency (Congruence).

        We treat the dataset as a dual-stream information source:
        1.  **The Symbol (Text):** What the script says (Semantic Score).
        2.  **The Signal (Audio):** How the actor performs it (Acoustic Score).

        By plotting these two normalized vectors against each other, we can identify:
        * **Congruence:** Points on the diagonal (Good Acting).
        * **Dissonance:** Points far from the diagonal (Irony, Subversion, or Bad Acting).
        """)

# ... inside Page 2 logic ...
elif page == "3. Analysis Part 1: The Style Tokens (Clustering)":
    st.title("Discovery: The 5 Style Tokens")
    st.markdown("We use **K-Means Clustering** to group the audio, and **PCA** to visualize the groups in 2D space.")

    # 1. RUN CLUSTERING
    kmeans = KMeans(n_clusters=5, random_state=42)
    df_raw['cluster'] = kmeans.fit_predict(df_norm[features])

    # 2. THE CLUSTER MAP (PCA Scatter Plot)
    # This is the "2D Chart" you were expecting!
    st.subheader("The Cluster Map")
    st.caption("We squash the 5 dimensions down to 2D to see how the groups separate.")

    # Run PCA to reduce 5 dimensions -> 2 dimensions
    pca = PCA(n_components=2)
    components = pca.fit_transform(df_norm[features])

    df_raw['pca_x'] = components[:, 0]
    df_raw['pca_y'] = components[:, 1]

    fig_map = px.scatter(
        df_raw,
        x='pca_x',
        y='pca_y',
        color=df_raw['cluster'].astype(str),
        title="The 5 Emotional Islands (PCA Projection)",
        hover_data=['text'],
        opacity=0.6
    )
    st.plotly_chart(fig_map, use_container_width=True)

    st.divider()

    # 3. THE STYLE TOKENS (Radar Charts)
    st.subheader("The 'Style Token' Signatures")
    st.markdown("Now we look at the **Average Shape** (Centroid) of each cluster to understand its vibe.")

    col1, col2 = st.columns([1, 2])

    with col1:
        cluster_id = st.selectbox("Select Token ID", sorted(df_raw['cluster'].unique()))
        subset = df_norm[df_raw['cluster'] == cluster_id]
        avg_stats = subset[features].mean()

        st.write(f"**Files:** {len(subset)}")
        st.json(avg_stats.to_dict())

    with col2:
        # Create Radar Chart
        categories = ['Loudness', 'Breathiness', 'Pitch', 'Brightness', 'Tonality']
        values = avg_stats.values.tolist()
        values += values[:1]

        fig = go.Figure(data=go.Scatterpolar(
            r=values,
            theta=categories + [categories[0]],
            fill='toself',
            name=f'Archetype {cluster_id}'
        ))
        fig.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0, 1])), showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    # 3. SAMPLES
    st.divider()
    st.write(f"**Listen to Archetype {cluster_id}:**")
    cols = st.columns(3)
    for i, (_, row) in enumerate(df_raw[df_raw['cluster'] == cluster_id].sample(3).iterrows()):
        with cols[i]:
            st.audio(row['path'])
            st.caption(row['text'])

# --- PAGE 3: INTIMACY (PHYSICS) ---
elif page == "4. Analysis Part 2: Engineering Intimacy (Physics)":
    st.title("The Physics of Intimacy")
    st.markdown(
        "We hypothesize that **Intimacy** and **Intensity** are opposing forces defined by the Proximity Effect.")

    # 1. CALCULATE SCORES
    # Intimacy = Breathy + Quiet
    df_raw['intimacy_score'] = (df_norm['breathiness'] * 1.5) + ((1 - df_norm['loudness']) * 1.0)
    # Intensity = Loud + High Pitch
    df_raw['intensity_score'] = (df_norm['loudness'] * 1.5) + (df_norm['pitch'] * 1.0)

    # 2. SCATTER PLOT
    st.subheader("Intimacy vs. Intensity Matrix")
    fig = px.scatter(
        df_raw,
        x="intimacy_score",
        y="intensity_score",
        color="breathiness",
        hover_data=['text'],
        title="The Emotional Landscape",
        labels={"intimacy_score": "Proximity (Soft)", "intensity_score": "Arousal (Hard)"},
        color_continuous_scale="Viridis"
    )
    # Add annotations
    fig.add_annotation(x=df_raw['intimacy_score'].max(), y=df_raw['intensity_score'].min(), text="ASMR Zone",
                       showarrow=False)
    fig.add_annotation(x=df_raw['intimacy_score'].min(), y=df_raw['intensity_score'].max(), text="Climax Zone",
                       showarrow=False)

    st.plotly_chart(fig, use_container_width=True)

# --- PAGE 4: CROSS-MODAL (SEMANTICS) ---
elif page == "5. Analysis Part 3: Cross-Modal Alignment (Text)":
    st.title("Cross-Modal Analysis")
    st.markdown("Do the words (Script) match the voice (Performance)?")

    # 1. NORMALIZE SCORES FOR COMPARISON
    # Scale Acoustic Intimacy to 0-10
    # Re-calculating purely for this view
    acoustic_raw = (df_norm['breathiness'] * 1.5) + ((1 - df_norm['loudness']) * 1.0)
    df_raw['acoustic_final'] = (acoustic_raw / acoustic_raw.max()) * 10

    # Scale Semantic Score to 0-10
    # Note: Prepare_data.py must have 'semantic_score' in the CSV!
    if 'semantic_score' in df_raw.columns:
        df_raw['semantic_final'] = (df_raw['semantic_score'] / df_raw['semantic_score'].max()) * 10
    else:
        st.error("Semantic Score missing. Check prepare_data.py")
        st.stop()

    # 2. THE ALIGNMENT MATRIX
    col1, col2 = st.columns([3, 1])

    with col1:
        fig = px.scatter(
            df_raw,
            x="semantic_final",
            y="acoustic_final",
            color="pitch",
            size='duration',
            hover_data=['text'],
            title="Performance Alignment: Script vs. Voice",
            labels={"semantic_final": "Script Intimacy (Text)", "acoustic_final": "Vocal Intimacy (Audio)"}
        )
        # The Line of Perfect Acting
        fig.add_shape(type="line", x0=0, y0=0, x1=10, y1=10, line=dict(color="Green", dash="dash"))
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.info("ℹ️ **Interpretation**")
        st.write("**On Line:** Perfect Acting (Voice matches Text)")
        st.write("**Top Left:** Subversion / ASMR (Boring text, Intimate voice)")
        st.write("**Bottom Right:** Bad Acting (Romantic text, Boring voice)")

    # 3. DRILL DOWN
    st.divider()
    st.subheader("Inspect Anomalies")
    anomaly_type = st.selectbox("Select Case",
                                ["High Script / Low Voice (Bad Acting?)", "Low Script / High Voice (ASMR?)"])

    if "Bad" in anomaly_type:
        subset = df_raw[(df_raw['semantic_final'] > 7) & (df_raw['acoustic_final'] < 4)]
    else:
        subset = df_raw[(df_raw['semantic_final'] < 3) & (df_raw['acoustic_final'] > 7)]

    st.write(f"Found {len(subset)} clips.")
    for i, (_, row) in enumerate(subset.head(3).iterrows()):
        st.audio(row['path'])
        st.write(f"**Script:** {row['text']}")
        st.caption(f"Text Score: {row['semantic_final']:.1f} | Audio Score: {row['acoustic_final']:.1f}")
        st.divider()