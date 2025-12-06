import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import os
import numpy as np
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

        # --- THE CRITICAL FIX FOR DEPLOYMENT ---
        # If running on Linux (Streamlit Cloud), Windows paths (\) will fail.
        # We replace all backslashes with forward slashes to make it cross-platform.
        if 'path' in df.columns:
            df['path'] = df['path'].astype(str).str.replace('\\', '/', regex=False)

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
    "3. Analysis & Insights",
    "4. Conclusions"
])

# --- GLOBAL DATA PREP (Normalize Once) ---
# We normalize features 0-1 for all analysis to keep math consistent
features = ['loudness', 'breathiness', 'pitch', 'brightness', 'tonality']
scaler = MinMaxScaler()
df_norm = df_raw.copy()
df_norm[features] = scaler.fit_transform(df_raw[features])

# --- NEW: INITIALIZE SESSION STATE FOR SLIDERS ---
if 'w_breath' not in st.session_state:
    st.session_state.w_breath = 1.5
if 'w_quiet' not in st.session_state:
    st.session_state.w_quiet = 1.0

# --- NEW: GLOBAL CALCULATION (Needed for Page 3 Tab 1) ---
# This ensures 'dynamic_intimacy' exists before you click the PCA map
df_raw['dynamic_intimacy'] = (
    df_norm['breathiness'] * st.session_state.w_breath) + (
    (1 - df_norm['loudness']) * st.session_state.w_quiet)

df_raw['dynamic_intensity'] = (df_norm['loudness'] * 1.5) + (df_norm['pitch'] * 1.0)

# --- PAGE 1: OVERVIEW ---
if page == "1. Project Overview":
    st.title("Engineering Intimacy: The Paralinguistics of Voice")
    st.markdown("### What it's about: Project Overview")
    # 1. THE HOOK (The Hybrid Approach)
    with st.expander("Abstract: Research Question & Hypothesis"):
        st.info("""
        **Research Question:** Can we mathematically "Engineer" the feeling of intimacy?
    
        **The Hypothesis:** Intimacy is not random. It is a specific **Paralinguistic Code** composed of breathiness, proximity, and rhythm. 
        By analyzing **1,000** clips of hyper-stylized voice acting, we aim to reverse-engineer this code to prove that "Emotion" is just a vector of numbers.
        """)

    # 2. THE TRANSITION (Why this dataset?)
    st.markdown("""
    **Why this dataset?** We selected the *NandemoGHS* dataset because it represents **"Performative Intimacy."** The voice actors are not just speaking; they are utilizing extreme paralinguistic control to simulate closeness. This makes it the perfect laboratory to study the **Proximity Effect**.
    """)

    st.markdown("### Why it Matters: The Prosody Transfer")
    st.markdown("""
    > "Current Text-to-Speech (TTS) models have mastered **clarity**, but they struggle with **performance**. They can read a script, but they cannot *act*. 
    >
    > This dataset provides a unique opportunity to solve the **'Prosody Gap'** in Generative Audio. By isolating extreme emotional states, from breathy intimacy to high-pitch panic, we can extract mathematical **'Style Tokens'** that decouple *emotion* from *identity*. 
    >
    > This analysis lays the groundwork for **Next-Gen Dubbing Technology** (e.g., YouTube Auto-Dub), where an AI must not only translate the words of a creator but also preserve their emotional intensity and pacing across language barriers. [Example](https://www.youtube.com/shorts/vR8ss6V0fHI)"
    """)

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

    st.subheader("Data Dictionary & Variable Definitions")

    # 1. ACOUSTIC FEATURES (The Physics)
    st.markdown("#### 1. Acoustic Features (The Signal)")
    st.caption("Extracted via `librosa` from raw audio waves.")

    acoustic_data = [
        {"Field": "duration", "Metric": "Time (Seconds)",
         "Definition": "Length of the clip. <1.0s usually indicates non-lexical sounds (gasps, moans)."},
        {"Field": "loudness", "Metric": "RMS (Energy)",
         "Definition": "Volume/Intensity. High = Shouting/Anger. Low = Whispering/Intimacy."},
        {"Field": "breathiness", "Metric": "Zero-Crossing Rate",
         "Definition": "Texture/Airiness. High ZCR indicates 'noise' (whispers). Low ZCR indicates clear tone."},
        {"Field": "pitch", "Metric": "Fundamental Freq (F0)",
         "Definition": "The Tone (Hz). High (>300Hz) = 'Moe'/Panic. Low (<220Hz) = Mature/Dominant."},
        {"Field": "brightness", "Metric": "Spectral Centroid",
         "Definition": "Timbre (Sharpness). High = Bright/Anime voice. Low = Dark/Chest voice."},
        {"Field": "tonality", "Metric": "1 - Spectral Flatness",
         "Definition": "Stability. 1.0 = Pure musical note. 0.0 = Shaky/Breathy noise."}
    ]
    st.table(pd.DataFrame(acoustic_data))

    # 2. SEMANTIC FEATURES (The Symbol)
    st.markdown("#### 2. Semantic Features (The Symbol)")
    st.caption("Derived from the text transcript.")

    semantic_data = [
        {"Field": "speed", "Metric": "Char / Duration",
         "Definition": "Information Density. Fast = Anxiety/Excitement. Slow = Seduction/Comfort."},
        {"Field": "semantic_score", "Metric": "Keyword Score (0-10)",
         "Definition": "Script Intimacy. Based on weighted keywords (e.g., 'Love'=10, 'Stop'=-5)."}
    ]
    st.table(pd.DataFrame(semantic_data))

    # 3. ENGINEERED METRICS (The Model)
    st.markdown("#### 3. Engineered Metrics (The Algorithms)")
    st.caption("Composite scores calculated using our custom formulas.")

    engineered_data = [
        {"Field": "dynamic_intimacy", "Formula": "Breathiness + (1 - Loudness)",
         "Definition": "The Proximity Effect. Measures how 'close' the speaker sounds."},
        {"Field": "dynamic_intensity", "Formula": "Loudness + Pitch",
         "Definition": "The Arousal Level. Measures the physical energy of the performance."}
    ]
    st.table(pd.DataFrame(engineered_data))


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
# ... inside app.py ...

elif page == "3. Analysis & Insights":
    st.title("Analysis & Insights")
    st.markdown("We utilized Unsupervised Machine Learning to uncover hidden patterns in the audio data.")
    # 1. Prepare Features
    features = ['loudness', 'breathiness', 'pitch', 'brightness', 'tonality']

    # 2. Run Clustering (K-Means)
    kmeans = KMeans(n_clusters=5, random_state=42)
    df_raw['cluster'] = kmeans.fit_predict(df_norm[features])

    # 3. Run PCA (The Map)
    pca = PCA(n_components=2)
    components = pca.fit_transform(df_norm[features])
    df_raw['pca_x'] = components[:, 0]
    df_raw['pca_y'] = components[:, 1]
    # --- TAB 1: THE CLUSTERS (STYLE TOKENS) ---
    tab1, tab2, tab3 = st.tabs(["The Style Tokens", "The Intimacy Formula", "Cross-Modal Alignment"])

    with tab1:
        st.subheader("1. The 5 Style Tokens (Clustering)")

        # --- INTERACTION: KEYWORD HIGHLIGHTER ---
        c_filter, c_info = st.columns([1, 2])
        with c_filter:
            search_term = st.text_input("🔍 Highlight Keyword (e.g., 'Love', 'Kiss')", "")

        # 1. CREATE THE MISSING COLUMN (THE FIX)
        # This copies the index into a real column so Plotly can see it
        df_raw['real_id'] = df_raw.index

        # 2. SETUP COLORS
        if search_term:
            df_raw['is_highlighted'] = df_raw['text'].str.contains(search_term, na=False, case=False)
            color_col = 'is_highlighted'
            title_text = f"PCA Map: Highlighting '{search_term}'"
            color_map = {True: "red", False: "lightgrey"}
        else:
            color_col = df_raw['cluster'].astype(str)
            title_text = "PCA Projection: The 5 Emotional Islands"
            color_map = None

        # A. CREATE THE PCA MAP FIGURE
        fig_map = px.scatter(
            df_raw,
            x='pca_x',
            y='pca_y',
            color=color_col,
            title=title_text,
            # Now 'real_id' exists, so this line won't crash!
            custom_data=['real_id'],
            hover_data=['text', 'cluster'],
            opacity=0.7,
            color_discrete_map=color_map,
            color_discrete_sequence=px.colors.qualitative.Bold
        )
        fig_map.update_layout(clickmode='event+select')

        # B. RENDER WITH EVENT LISTENER
        event = st.plotly_chart(
            fig_map,
            on_select="rerun",
            selection_mode="points",
            use_container_width=True
        )

        # C. HANDLE CLICK EVENT
        if len(event["selection"]["points"]) > 0:
            # Get the "Passport" ID we stuffed into custom_data
            clicked_real_id = event["selection"]["points"][0]["customdata"][0]

            # Lookup the row
            selected_row = df_raw.loc[clicked_real_id]

            # Display Audio
            st.info(f"▶️ **Selected Clip:** {selected_row['text']}")
            st.caption("🎧 **Headphones Recommended for a better experience**")
            c_audio, c_meta = st.columns([1, 2])
            with c_audio:
                st.audio(selected_row['path'])
            with c_meta:
                st.caption(f"**Cluster:** {selected_row['cluster']}")
                if 'dynamic_intimacy' in selected_row:
                    st.caption(f"**Intimacy Score:** {selected_row['dynamic_intimacy']:.2f}")

        # 3. THE STYLE TOKENS (Radar Charts)
        cluster_insights = {
            0: {
                "title": "The Mature / Cool Type (Kuudere)",
                "desc": """
                        **The 'Stability' Token.**
                        * **Signature:** Low Pitch (0.21) + Max Tonality (0.85).
                        * **Vibe:** Composed, serious, and unwavering. Unlike the 'Panic' voices, this archetype has zero jitter. It represents the 'Cool Beauty' or narrator archetype.
                        """
            },
            1: {
                "title": "The Standard Heroine (Neutral)",
                "desc": """
                        **The 'Neutrality' Token.**
                        * **Signature:** Balanced Pitch (0.47) + High Tonality.
                        * **Vibe:** Clear and professional. This is the 'Control Group' of the dataset—standard dialogue without extreme affection (breathiness) or aggression (loudness).
                        """
            },
            2: {
                "title": "The Sweet / Imouto Type (Deredere)",
                "desc": """
                        **The 'Affection' Token.**
                        * **Signature:** Low Loudness (0.33) + Elevated Breathiness.
                        * **Vibe:** Soft-spoken and safe. The high 'Brightness' suggests a smaller vocal tract (younger character), while the quiet delivery engineers a sense of vulnerability and comfort.
                        """
            },
            3: {
                "title": "The Dominant / Sadist Type (S-Type)",
                "desc": """
                        **The 'Dominance' Token.**
                        * **Signature:** Lowest Breathiness (0.17) + High Tonality.
                        * **Vibe:** Hard, direct, and piercing. The complete lack of 'air' in the voice removes any sense of intimacy, creating a commanding or scolding tone used by strict characters.
                        """
            },
            4: {
                "title": "The Seductive / Onee-san Type",
                "desc": """
                        **The 'Seduction' Token.**
                        * **Signature:** Lowest Pitch (0.18) + High Brightness (0.65).
                        * **Vibe:** Deep intimacy. This unique combo (Low Pitch + High Texture) indicates **Vocal Fry**—the gravelly, mature register used to signal deep, overwhelming closeness.
                        """
            }
        }

        st.subheader("The 'Style Token' Signatures")
        st.markdown("Now we look at the **Average Shape** (Centroid) of each cluster to understand its vibe.")

        col1, col2 = st.columns([1, 2])

        with col1:
            cluster_id = st.selectbox("Select Token ID", sorted(df_raw['cluster'].unique()))
            subset = df_norm[df_raw['cluster'] == cluster_id]
            avg_stats = subset[features].mean()

            st.write(f"**Files:** {len(subset)}")
            st.json(avg_stats.to_dict())

        insight = cluster_insights.get(cluster_id,
                                       {"title": f"Cluster {cluster_id}",
                                        "desc": "No manual analysis available."})

        # Display Text
        st.success(f"**{insight['title']}**")
        st.markdown(insight['desc'])
        st.caption(f"Based on {len(subset)} audio samples.")

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
    # --- TAB 2: INTIMACY ENGINEERING ---
    with tab2:
        st.subheader("2. Engineering Intimacy (The Physics)")

        # Filter Logic
        min_dur = st.slider("Hide clips shorter than (seconds):", 0.0, 5.0, 1.0, step=0.5)

        # We perform the filter, but we KEEP 'real_id' so the click still works!
        filtered_df = df_raw[df_raw['duration'] >= min_dur].copy()

        st.caption(f"Showing {len(filtered_df)} clips")

        # Formula Tuners
        with st.expander("🎛️ Tune the Intimacy Formula"):
            w_breath = st.slider("Weight: Breathiness", 0.5, 2.0, 1.5)
            w_quiet = st.slider("Weight: Quietness", 0.5, 2.0, 1.0)

        # Recalculate (Using normalized subset)
        norm_subset = df_norm.loc[filtered_df.index]
        filtered_df['dynamic_intimacy'] = (norm_subset['breathiness'] * w_breath) + (
                    (1 - norm_subset['loudness']) * w_quiet)
        filtered_df['dynamic_intensity'] = (norm_subset['loudness'] * 1.5) + (norm_subset['pitch'] * 1.0)

        # A. INTERACTIVE SCATTER PLOT
        fig_scat = px.scatter(
            filtered_df,
            x="dynamic_intimacy",
            y="dynamic_intensity",
            color="breathiness",
            title="The Emotional Landscape (Select a point to listen)",
            # CRITICAL: Pass the ID so we can find the audio
            custom_data=['real_id'],
            hover_data=['text'],
            labels={"dynamic_intimacy": "Proximity (Intimacy)", "dynamic_intensity": "Arousal (Intensity)"},
            color_continuous_scale="Viridis"
        )
        fig_scat.update_layout(clickmode='event+select')

        # Add the Green Zone
        fig_scat.add_shape(type="rect", x0=1.5, y0=0, x1=2.5, y1=1.0, line=dict(color="Green"), fillcolor="Green",
                           opacity=0.1)

        # Render Chart
        event_2 = st.plotly_chart(fig_scat, on_select="rerun", selection_mode="points", use_container_width=True)

        # B. CLICK HANDLER (TAB 2)
        if len(event_2["selection"]["points"]) > 0:
            clicked_id = event_2["selection"]["points"][0]["customdata"][0]
            row = df_raw.loc[clicked_id]

            st.info(f"▶️ **Selected Analysis:** {row['text']}")
            st.caption("🎧 **Headphones Recommended for a better experience**")

            c1, c2 = st.columns([1, 2])
            with c1: st.audio(row['path'])
            with c2:
                st.metric("Intimacy Score", f"{row['dynamic_intimacy']:.2f}")
                st.metric("Intensity Score", f"{row['dynamic_intensity']:.2f}")

        # D. VIDEO REFERENCE
        st.divider()
        st.subheader("🎥 Examples: Intimacy vs Arousal")

        # UPDATE THIS PATH to your actual video file
        VIDEO_PATH_1 = "intimacy.mp4"
        VIDEO_PATH_2 = "arousal.mp4"

        # Create two columns for side-by-side layout
        col1, col2 = st.columns(2)

        with col1:
            if os.path.exists(VIDEO_PATH_1):
                st.caption("Reference material for paralinguistic intimacy.")
                st.video(VIDEO_PATH_1)
            else:
                st.warning(f"Video reference not found at: {VIDEO_PATH_1}")

        with col2:
            if os.path.exists(VIDEO_PATH_2):
                st.caption("Reference material for paralinguistic arousal.")
                st.video(VIDEO_PATH_2)
            else:
                st.warning(f"Video reference not found at: {VIDEO_PATH_2}")
            # --- THE DISCOVERY BLOCK ---
        st.divider()
        st.subheader("Insight: The Biological Constraint")
        st.info("""
           **Initially, we hypothesized that Intimacy and Intensity were orthogonal vectors**, similar to the Russell Circumplex Model, implying they could vary independently.

           **However, our data refutes this.** The scatter plot reveals a strong **Negative Correlation**. This led to a key discovery: **The Biological Constraint of Intimacy.**

           The "Empty Top-Right Corner" of the graph proves that high-arousal intimacy is physically impossible. To maximize the *Proximity Effect* (Intimacy), an actor must sacrifice *Intensity*. 

           This suggests that in the grammar of voice acting, **Vulnerability (Breathiness) requires the surrender of Power (Loudness).**
           """)

    # --- TAB 3: CROSS-MODAL ALIGNMENT ---
    with tab3:
        st.subheader("3. Cross-Modal Alignment (The Acting Check)")

        st.markdown("**Find Anomalies**")
        threshold = st.slider("Mismatch Severity Threshold", 0.0, 5.0, 0.0)

        # Calculate Logic
        acoustic_raw = (df_norm['breathiness'] * 1.5) + ((1 - df_norm['loudness']) * 1.0)
        df_raw['acoustic_final'] = (acoustic_raw / acoustic_raw.max()) * 10
        if 'semantic_score' in df_raw.columns:
            df_raw['semantic_final'] = (df_raw['semantic_score'] / df_raw['semantic_score'].max()) * 10

        df_raw['mismatch'] = abs(df_raw['acoustic_final'] - df_raw['semantic_final'])
        anomaly_df = df_raw[df_raw['mismatch'] >= threshold].copy()

        # A. INTERACTIVE SCATTER PLOT
        fig_align = px.scatter(
            anomaly_df,
            x="semantic_final",
            y="acoustic_final",
            color="mismatch",
            size='duration',
            title=f"Performance Alignment ({len(anomaly_df)} clips)",
            # CRITICAL: Pass the ID
            custom_data=['real_id'],
            hover_data=['text'],
            labels={"semantic_final": "Script Intimacy (0-10)", "acoustic_final": "Vocal Intimacy (0-10)"},
            color_continuous_scale="Turbo"
        )
        fig_align.add_shape(type="line", x0=0, y0=0, x1=10, y1=10, line=dict(color="Green", dash="dash"))
        fig_align.update_layout(clickmode='event+select')

        event_3 = st.plotly_chart(fig_align, on_select="rerun", selection_mode="points", use_container_width=True)

        # B. CLICK HANDLER (TAB 3)
        if len(event_3["selection"]["points"]) > 0:
            clicked_id = event_3["selection"]["points"][0]["customdata"][0]
            row = df_raw.loc[clicked_id]

            st.info(f"▶️ **Selected Anomaly:** {row['text']}")
            st.caption("🎧 **Headphones Recommended for a better experience**")

            c1, c2 = st.columns([1, 2])
            with c1:
                st.audio(row['path'])
            with c2:
                st.write(f"**Mismatch Score:** {row['mismatch']:.1f}")
                if row['acoustic_final'] > row['semantic_final']:
                    st.success("Type: **Elevation** (Voice > Text)")
                else:
                    st.error("Type: **Subversion/Bad Acting** (Text > Voice)")
        st.info(f"""
       Our Cross-Modal Analysis reveals that while most performances align with the script (the diagonal), the most interesting data points are the Dissonant Outliers.

The massive outlier in the top-left (Score: 0 Text / 10 Audio) proves that in this genre, Paralinguistics override Semantics. The actors are injecting massive amounts of intimacy into neutral text, effectively 'elevating' the script through pure vocal technique. Conversely, the outliers on the bottom-right likely capture 'Tsundere' performances, where the acoustic anger masks the semantic affection.
        """)

# --- PAGE 3: INTIMACY (PHYSICS) ---

# ... inside app.py ...

elif page == "4. Conclusions":
    st.title("Conclusions & Recommendations")

    # 1. THE EXECUTIVE SUMMARY
    st.markdown("""
    ### The "Paralinguistic Code" Decoded
    Our research set out to determine if the "soul" of a voice actor could be quantified. 
    By applying unsupervised learning to 1,000 audio samples, we were able to reverse-engineered the **Acoustic DNA of Emotion**.
    """)

    # 2. VISUAL SUMMARY OF FINDINGS
    col1, col2 = st.columns(2)

    with col1:
        st.info("**The 5 Style Tokens (Clustering)**")
        st.markdown("""
        * **Cluster 0:** Stability (Low Pitch, Pure Tone)
        * **Cluster 1:** Neutrality (Balanced, Clear)
        * **Cluster 2:** Affection (Quiet, Breathy)
        * **Cluster 3:** Dominance (Hard, Direct)
        * **Cluster 4:** Seduction (Deep, Vocal Fry)
        """)

    with col2:
        st.success("**The Physics of Intimacy**")
        st.markdown("""
        * **The Discovery:** Intimacy and Intensity are **Mutually Exclusive**.
        * **The Constraint:** You cannot scream in a whisper. 
        * **The Metric:** **Breathiness** is the currency of closeness.
        """)
    st.error("**Cross-Modal Analysis**")
    st.markdown("""
    * The massive outlier in the top-left (Score: 0 Text / 10 Audio) proves that in this genre.
    * Paralinguistics override Semantics
    """)

    st.divider()

    # ... inside Page 4 ...

    # 3. INTERACTIVE: THE SCENARIO PLANNER
    st.subheader("Interactive: Style Transfer Director")
    st.markdown("Simulate how these 'Style Tokens' would be applied in a commercial TTS pipeline.")

    # INPUTS
    c1, c2 = st.columns([1, 1])
    with c1:
        # THE DROPDOWN (Updated for Tech Applications)
        target_audience = st.selectbox(
            "1. Select Application Case",
            ["YouTube Auto-Dub (High Energy)", "Audiobook (Romance)", "NPC Dialogue (Boss Fight)",
             "GPS Navigation (Calm)"]
        )
    with c2:
        sample_line = st.text_input(
            "2. Enter Sample Script Line",
            "I can't believe we actually made it this far!"
        )

    # LOGIC ENGINE (Updated)
    if target_audience == "YouTube Auto-Dub (High Energy)":
        rec_cluster = "Cluster 0/3 (Stability/Dominant)"
        tuning = "Maximize Loudness, High Variance in Pitch."
        reason = "Matches the 'Hype' energy of creators like MrBeast. Needs high dynamic range to retain viewer retention."
    elif target_audience == "Audiobook (Romance)":
        rec_cluster = "Cluster 4 (Deep)"
        tuning = "Lower Pitch, Increase Vocal Fry (Texture)."
        reason = "Creates an immersive, 'in-ear' storytelling experience closer to radio drama than standard narration."
    elif target_audience == "NPC Dialogue (Boss Fight)":
        rec_cluster = "Cluster 3 (Dominance)"
        tuning = "Maximize Tonality, Zero Breathiness, Hard Attack."
        reason = "Cut through game sound effects (SFX). Signals authority and threat level to the player."
    else:  # GPS Navigation
        rec_cluster = "Cluster 1 (Neutral)"
        tuning = "Maximize Stability, Normalize Pitch."
        reason = "Cognitive Load Management. The driver needs information without emotional distraction."

    # OUTPUT CARD
    st.warning(f"""
        **Use Case:** {target_audience}
        **Script:** *"{sample_line}"*

        ---
        **Target Style Token:** {rec_cluster}
        **Prosody Tuning:** {tuning}
        **Engineering Logic:** {reason}
        """)

    st.divider()

    # 4. LIMITATIONS & CHALLENGES (Inserted Here)
    st.subheader("4. Project Limitations & Challenges")
    st.markdown(
        "While our model successfully identified broad archetypes, we encountered specific challenges inherent to the dataset.")

    with st.expander("⚠️ Read Limitation Analysis"):
        st.markdown("""
        **1. The Global Averaging Bias (The 'Dilution' Effect)**
        * **Challenge:** Our model calculates the average acoustic profile of an entire clip.
        * **Implication:** Mixed-emotion clips are misclassified. If a character shouts a command (High Energy) and then sighs (High Breathiness) in the same file, the shout "dilutes" the breathiness score. This causes the file to be sorted into **Cluster 3 (Dominance)** instead of **Cluster 2 (Affection)**, proving the need for millisecond-level Time-Series Analysis.

        **2. Acoustic Ambiguity (The 'Tonal Moan' Problem)**
        * **Challenge:** We found moans in **Cluster 0 (Stability)**, which was unexpected.
        * **Cause:** To the algorithm, a deep, resonant moan (*"Mmm..."*) looks mathematically identical to a serious narration. Both are **Low Pitch** and **High Tonality** (Stable). The model failed to capture the eroticism because the actor used *Resonance* instead of *Breathiness*, revealing a conflict between **Semantic Category** (Moan) and **Acoustic Physics** (Stable Tone).

        **3. Temporal Instability (The 'Short Clip' Problem)**
        * **Challenge:** Clips shorter than 1.0 second (e.g., gasps) often produced chaotic spectral data.
        * **Cause:** Features like Pitch ($F_0$) require sustained vibration to stabilize. Short bursts appear as "noise," leading to outliers in the Clustering Map.

        **4. Semantic Blindness (Irony & Subtext)**
        * **Challenge:** The Cross-Modal Alignment flagged "Tsundere" lines as "Bad Acting."
        * **Cause:** Our Rule-Based NLP relies on keywords. It cannot detect **Irony** (saying "I hate you" affectionately).

        **5. Ecological Validity (Performative Bias)**
        * **Challenge:** The "Intimacy" engineered here is *Performative*.
        * **Implication:** The acoustic features are exaggerated (hyper-breathy, hyper-high pitch). Models trained on this data risk falling into the "Uncanny Valley" if deployed for real-world interactions.
            """)

    # 5. FINAL RELEVANCE STATEMENT (The "So What?")
    st.markdown("### Impact Statement: Expressive Performance Transfer")
    st.info("""
         "The primary relevance of this project lies in bridging the **expressivity gap** in modern AI voice synthesis (TTS). 
         Systems like YouTube's auto-dub are transitioning from simple language translation to **performance transfer**. 
         Our analysis of the **5 Style Tokens** provides the essential framework for this technology. These clusters function as **Latent Vectors**—the exact acoustic recipes required to inject emotion and maintain a consistent **speaker style** across different languages."
        """)