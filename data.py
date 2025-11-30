import os
import pandas as pd
from datasets import load_dataset, Audio
import soundfile as sf
import librosa
import numpy as np
import io
from tqdm import tqdm

# CONFIG
DATASET_NAME = "NandemoGHS/Japanese-Eroge-Voice"
SAMPLE_SIZE = 500  # 500 items is a good sample size for K-Means
OUTPUT_DIR = "final_project_data"
AUDIO_DIR = os.path.join(OUTPUT_DIR, "audio")
CSV_PATH = os.path.join(OUTPUT_DIR, "features.csv")

os.makedirs(AUDIO_DIR, exist_ok=True)

# LOAD STREAM
print("🔌 Connecting to Hugging Face...")
ds = load_dataset(DATASET_NAME, split="train", streaming=True)
ds = ds.cast_column("flac", Audio(decode=False))

data_records = []

print(f"⬇️ Processing {SAMPLE_SIZE} files for analysis...")
iterable = ds.take(SAMPLE_SIZE)

for i, item in tqdm(enumerate(iterable), total=SAMPLE_SIZE):
    try:
        # 1. Save Audio File
        filename = f"voice_{i:04d}.wav"
        filepath = os.path.join(AUDIO_DIR, filename)

        audio_bytes = item['flac']['bytes']
        virtual_file = io.BytesIO(audio_bytes)
        y, sr = sf.read(virtual_file)
        sf.write(filepath, y, sr)

        # 2. FEATURE EXTRACTION (The Data Analysis Part!)
        # This converts sound into numbers we can analyze.

        # Load back with librosa for analysis features
        y_librosa, sr_librosa = librosa.load(filepath, sr=None)

        # Feature 1: Duration
        duration = librosa.get_duration(y=y_librosa, sr=sr_librosa)

        # Feature 2: RMS (Loudness)
        rms = np.mean(librosa.feature.rms(y=y_librosa))

        # Feature 3: Zero Crossing Rate (Roughness/Noise)
        zcr = np.mean(librosa.feature.zero_crossing_rate(y=y_librosa))

        # Feature 4: Spectral Centroid (Brightness of sound)
        centroid = np.mean(librosa.feature.spectral_centroid(y=y_librosa, sr=sr_librosa))

        data_records.append({
            "id": i,
            "filename": filename,
            "text": item.get('txt', ''),
            "duration": duration,
            "loudness": rms,
            "roughness": zcr,
            "brightness": centroid,
            "path": filepath  # Relative path for Streamlit
        })

    except Exception as e:
        print(f"Skipping {i} due to error: {e}")

# Save the tabular data
df = pd.DataFrame(data_records)
df.to_csv(CSV_PATH, index=False)
print(f"✅ Success! Data saved to {CSV_PATH}")