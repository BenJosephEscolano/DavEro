import os
import pandas as pd
from datasets import load_dataset, Audio
import soundfile as sf
import librosa
import numpy as np
import io
from tqdm import tqdm

# --- CONFIGURATION ---
DATASET_NAME = "NandemoGHS/Japanese-Eroge-Voice"
SAMPLE_SIZE = 1000  # 500 is the sweet spot for a classroom demo
OUTPUT_DIR = "final_project_data"
AUDIO_DIR = os.path.join(OUTPUT_DIR, "audio")
CSV_PATH = os.path.join(OUTPUT_DIR, "multimodal_features.csv")

# Create folders if they don't exist
os.makedirs(AUDIO_DIR, exist_ok=True)

# --- SEMANTIC ANALYSIS (The "Consumption" Method) ---

# 1. CATEGORIZED DICTIONARY
# Note: We include "Trap Words" like Kawaii to prevent false positives.
LEXICON = {
    # --- TRAP WORDS (Consume these first so they don't trigger others) ---
    "可愛い": 2,  # Kawaii (Cute) - Contains 'Love', so we catch it here.
    "可愛": 2,  # Kawaii (stem)
    "やめないで": 8,  # Yamenaide (Don't stop) - Catch before "Yamete" (Stop)

    # --- HIGH INTENSITY (Long phrases) ---
    "大好き": 10,  # Daisuki (Love)
    "愛してる": 10,  # Aishiteru (Deep Love)
    "気持ちいい": 5,  # Kimochii (Feels good)

    # --- MEDIUM INTENSITY ---
    "好き": 5,  # Suki (Like)
    "幸せ": 5,  # Shiawase (Happy)
    "一緒": 4,  # Issho (Together)
    "キス": 6,  # Kisu (Kiss)
    "抱いて": 7,  # Daite (Hold me)
    "触って": 5,  # Sawatte (Touch me)
    "近い": 4,  # Chikai (Close)
    "奥": 6,  # Oku (Deep inside)
    "もっと": 4,  # Motto (More)
    "すごい": 3,  # Sugoi (Amazing)

    # --- LOW INTENSITY / SHORT ---
    "愛": 8,  # Ai (Love) - Only triggers if 'Aishiteru' wasn't found
    "声": 3,  # Voice
    "耳": 4,  # Ear
    "あっ": 2, "んっ": 2, "ぁ": 1,  # Breath/Moans

    # --- NEGATIVE (Stop words) ---
    "やだ": -5,  # Yada (No/Stop)
    "やめて": -5,  # Yamete (Stop)
}


def get_semantic_score(text):
    if not isinstance(text, str): return 0

    score = 0
    temp_text = text  # Work on a copy of the text

    # 1. SORT KEYS BY LENGTH (Longest first)
    # This ensures "愛してる" (5 chars) is checked before "愛" (1 char)
    sorted_keys = sorted(LEXICON.keys(), key=len, reverse=True)

    # 2. CONSUMPTION LOOP
    for word in sorted_keys:
        if word in temp_text:
            # How many times does it appear?
            count = temp_text.count(word)

            # Add points
            points = LEXICON[word]
            score += (points * count)

            # REMOVE the word from the text so it can't be matched again
            # We replace it with a blank space to prevent merging remaining chars
            temp_text = temp_text.replace(word, " ")

    return score

# --- CONNECT TO DATASTREAM ---
print("🔌 Connecting to Hugging Face...")
# We use streaming=True so we don't download 100GB
ds = load_dataset(DATASET_NAME, split="train", streaming=True)
# Force decode=False to bypass the 'torchcodec' error on Windows
ds = ds.cast_column("flac", Audio(decode=False))

data_records = []
iterable = ds.take(SAMPLE_SIZE)

print(f"⬇️ Downloading and Analyzing {SAMPLE_SIZE} files...")
print("This may take 5-10 minutes depending on your internet/CPU.")

for i, item in tqdm(enumerate(iterable), total=SAMPLE_SIZE):
    try:
        # 1. SAVE AUDIO FILE LOCALLY
        # Streamlit needs a real file path to play audio
        filename = f"voice_{i:04d}.wav"
        filepath = os.path.join(AUDIO_DIR, filename)

        # Extract raw bytes from Hugging Face
        audio_bytes = item['flac']['bytes']
        virtual_file = io.BytesIO(audio_bytes)

        # Read bytes -> Write WAV to disk
        y_raw, sr_raw = sf.read(virtual_file)
        sf.write(filepath, y_raw, sr_raw)

        # 2. LOAD WITH LIBROSA (For Analysis)
        # We reload the saved file to ensure format compatibility
        y, sr = librosa.load(filepath, sr=None)

        # --- FEATURE EXTRACTION ---

        # A. LOUDNESS (RMS) - Energy
        rms = np.mean(librosa.feature.rms(y=y))

        # B. BREATHINESS (Zero Crossing Rate) - Texture
        zcr = np.mean(librosa.feature.zero_crossing_rate(y=y))

        # C. PITCH (Fundamental Frequency / F0) - Tone
        # fmin=60, fmax=1000 covers the range of anime voice acting
        f0 = librosa.yin(y, fmin=60, fmax=1000)
        f0 = f0[f0 > 0]  # Remove silence (0 values)
        avg_pitch = np.mean(f0) if len(f0) > 0 else 0

        # D. BRIGHTNESS (Spectral Centroid) - Timbre
        # High centroid = "Sharp/Bright/Cute". Low = "Dark/Mature".
        centroid = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))

        # E. TONALITY (Spectral Flatness) - Stability
        # Flatness is 1.0 for white noise, 0.0 for a pure sine wave.
        # We invert it: 1.0 = Pure Tone, 0.0 = Noisy/Breathy
        flatness = np.mean(librosa.feature.spectral_flatness(y=y))
        tonality = 1.0 - flatness

        # F. SPEED (Semantics)
        text = item.get('txt', '')
        duration = librosa.get_duration(y=y, sr=sr)
        char_count = len(text)
        speed = char_count / duration if duration > 0 else 0

        # G. SEMANTIC SCORE (Keywords)
        sem_score = get_semantic_score(text)

        # 3. APPEND TO LIST
        data_records.append({
            "id": i,
            "filename": filename,
            "path": filepath,  # This path is used by st.audio()
            "text": text,
            "duration": duration,
            # Acoustic Features
            "loudness": rms,
            "breathiness": zcr,
            "pitch": avg_pitch,
            "brightness": centroid,
            "tonality": tonality,
            "speed": speed,
            # Semantic Features
            "semantic_score": sem_score
        })

    except Exception as e:
        # If a file is corrupt or pitch detection fails, just skip it
        # print(f"Skipped {i}: {e}")
        pass

# --- SAVE TO CSV ---
df = pd.DataFrame(data_records)

# Filter out rows where pitch detection failed (avg_pitch == 0)
df = df[df['pitch'] > 0]

df.to_csv(CSV_PATH, index=False)
print("-" * 30)
print(f"✅ SUCCESS! Processed {len(df)} files.")
print(f"📊 Data saved to: {CSV_PATH}")
print(f"🎧 Audio saved to: {AUDIO_DIR}")
print("-" * 30)
print("👉 Now run: streamlit run app.py")