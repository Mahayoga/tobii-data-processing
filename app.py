from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import numpy as np
from seaborn import heatmap
import random
import joblib
from collections import Counter

# ==========================================
# CONFIG
# ==========================================

DATASET_FOLDER = "datasets"

SCREEN_WIDTH = 1920
SCREEN_HEIGHT = 1080

GRID_SIZE = 3

FIXATION_THRESHOLD = 0.02
WINDOW_SECONDS = 10
SAMPLING_RATE = 3

WINDOW_SIZE = WINDOW_SECONDS * SAMPLING_RATE

SCREEN_W, SCREEN_H = None, None
app = FastAPI()

# Biar bisa diakses dari frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ==========================================
# LOAD MODEL
# ==========================================
feature_names = ['avg_velocity', 'max_velocity', 'std_velocity', 'total_distance', 'fixation_ratio', 'unique_area_count']
# Load model dan scaler
model = joblib.load('model/rfc_anxiety_model.pkl')
scaler = joblib.load('model/rfc_anxiety_scaler.pkl')

# ==========================================
# AREA GRID
# ==========================================

def get_area(x, y):

    col = int(x // (SCREEN_WIDTH / GRID_SIZE))
    row = int(y // (SCREEN_HEIGHT / GRID_SIZE))

    return f"{row}_{col}"

# ==========================================
# FEATURE EXTRACTION
# ==========================================

def extract_features_from_web(data):

    df = pd.DataFrame(data)

    # ambil hanya gaze valid
    df = df[['gaze_x', 'gaze_y', 'timestamp']]

    # ======================================
    # NORMALIZATION
    # ======================================

    df['gaze_x'] = df['gaze_x'] / SCREEN_WIDTH
    df['gaze_y'] = df['gaze_y'] / SCREEN_HEIGHT

    # ======================================
    # DELTA
    # ======================================

    df['dx'] = df['gaze_x'].diff()
    df['dy'] = df['gaze_y'].diff()

    df['dt'] = df['timestamp'].diff() / 1000

    # hapus NaN
    df = df.dropna()

    # ======================================
    # DISTANCE
    # ======================================

    df['distance'] = np.sqrt(
        df['dx']**2 +
        df['dy']**2
    )

    # ======================================
    # VELOCITY
    # ======================================

    df['velocity'] = df['distance'] / df['dt']

    # hapus infinite
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna()

    # ======================================
    # FIXATION
    # ======================================

    df['fixation'] = df['velocity'] < FIXATION_THRESHOLD

    # ======================================
    # AREA
    # ======================================

    df['area'] = df.apply(
        lambda row: get_area(
            row['gaze_x'] * SCREEN_WIDTH,
            row['gaze_y'] * SCREEN_HEIGHT
        ),
        axis=1
    )

    # ======================================
    # FEATURES
    # ======================================

    all_window_features = []

    for start in range(0, len(df), WINDOW_SIZE):

        window_df = df.iloc[start:start + WINDOW_SIZE]

        # skip kalau terlalu sedikit
        if len(window_df) < WINDOW_SIZE // 2:
            continue

        features = [

            # "avg_velocity":
                window_df['velocity'].mean(),

            # "max_velocity":
                window_df['velocity'].max(),

            # "std_velocity":
                window_df['velocity'].std(),

            # "total_distance":
                window_df['distance'].sum(),

            # "fixation_ratio":
                window_df['fixation'].mean(),

            # "unique_area_count":
                window_df['area'].nunique()
        ]

        all_window_features.append(features)

    return all_window_features

@app.post("/extract")
async def extract_features(request: Request):
    raw = await request.json()

    # handle format fleksibel
    if isinstance(raw, dict):
        SCREEN_W = int(raw.get("screen_w", 0))
        SCREEN_H = int(raw.get("screen_h", 0))
        raw = raw.get("data", [])
    elif isinstance(raw, list):
        print(f"Received list of {raw}")
        return {"error": "Expected a JSON object key."}
    else:
        print(f"Received unexpected data format: {raw}")
        return {"error": "Invalid data format."}

    df = pd.DataFrame(raw)

    # =============================
    # 🔧 PREPROCESSING
    # =============================
    df = df.sort_values(by="timestamp").reset_index(drop=True)

    # normalisasi (optional, sesuaikan resolusi)
    # SCREEN_W, SCREEN_H = 1920, 1080
    df['x'] = df['gaze_x'] / int(SCREEN_W)
    df['y'] = df['gaze_y'] / int(SCREEN_H)

    df['dx'] = df['x'].diff()
    df['dy'] = df['y'].diff()
    df['dt'] = df['timestamp'].diff() / 1000

    df = df.fillna(0)

    # =============================
    # 🔥 BUBBLE GRID
    # =============================
    # GRID_SIZE = 30  # makin besar makin detail
    # df['x_norm'] = df['gaze_x'] / SCREEN_W
    # df['y_norm'] = df['gaze_y'] / SCREEN_H

    # # mapping ke grid
    # df['x_bin'] = (df['x_norm'] * GRID_SIZE).astype(int)
    # df['y_bin'] = (df['y_norm'] * GRID_SIZE).astype(int)

    # # hitung frekuensi
    # bubble_grid = df.groupby(['x_bin', 'y_bin']).agg({
    #     'gaze_x': 'mean',
    #     'gaze_y': 'mean',
    #     'timestamp': 'count'
    # }).reset_index()
    # bubble_grid.rename(columns={'timestamp': 'count'}, inplace=True)

    # # cari max buat scaling radius
    # max_count = bubble_grid['count'].max()

    # convert ke format heatmap
    heatmap_data = []
    for _, row in df.iterrows():
        heatmap_data.append({
            "x": int(row['gaze_x']),
            "y": int(row['gaze_y']),
            "value": 1
        })

    # =============================
    # 📏 DISTANCE
    # =============================
    df['distance'] = np.sqrt(df['dx']**2 + df['dy']**2)

    # =============================
    # ⚡ VELOCITY & DISTANCE
    # =============================
    df['velocity'] = np.sqrt(df['dx']**2 + df['dy']**2) / (df['dt'] + 1e-6)
    df['distance'] = np.sqrt(df['dx']**2 + df['dy']**2)

    # =============================
    # 👀 FIXATION DETECTION
    # =============================
    FIX_THRESHOLD = 0.02
    df['fixation'] = df['velocity'] < FIX_THRESHOLD

    # =============================
    # 📊 GLOBAL FEATURES
    # =============================
    features = {
        "velocity_mean": float(df['velocity'].mean()),
        "velocity_std": float(df['velocity'].std()),
        "distance_total": float(df['distance'].sum()),
        "fixation_ratio": float(df['fixation'].mean()),
        "max_velocity": float(df['velocity'].max())
    }

    # =============================
    # 📊 STATISTICS
    # =============================
    avg_velocity = float(df['velocity'].mean())

    fixation_ratio = float(df['fixation'].mean() * 100)

    total_distance = float(df['distance'].sum())

    # =============================
    # 🎯 MOST VIEWED AREA
    # =============================

    def get_area(x, y):
        if x < SCREEN_W/3:
            h = "Kiri"
        elif x < 2*SCREEN_W/3:
            h = "Tengah"
        else:
            h = "Kanan"

        if y < SCREEN_H/3:
            v = "Atas"
        elif y < 2*SCREEN_H/3:
            v = "Tengah"
        else:
            v = "Bawah"

        return f"{h} {v}"

    df['area'] = df.apply(
        lambda row: get_area(row['gaze_x'], row['gaze_y']),
        axis=1
    )

    most_viewed_area = df['area'].mode()[0]

    # =============================
    # 🔥 WINDOWING (5 detik)
    # =============================
    WINDOW_SIZE = 5  # detik
    windows = []

    start_time = df['timestamp'].min()
    end_time = df['timestamp'].max()

    current = start_time

    while current < end_time:
        w = df[(df['timestamp'] >= current) & 
               (df['timestamp'] < current + WINDOW_SIZE * 1000)]

        if len(w) > 5:
            windows.append({
                "velocity_mean": float(w['velocity'].mean()),
                "velocity_std": float(w['velocity'].std()),
                "fixation_ratio": float(w['fixation'].mean()),
                "distance_total": float(w['distance'].sum())
            })

        current += WINDOW_SIZE * 1000

    MAX_POINTS = 300
    df_down = df.copy()

    if len(df) > MAX_POINTS:
        chunk_size = len(df) // MAX_POINTS
        df_down = df.groupby(np.arange(len(df)) // chunk_size).agg({
            'timestamp': 'mean',
            'velocity': 'mean'
        })

    # =============================
    # 📤 RESPONSE
    # =============================
    return {
        "global_features": features,
        "window_features": windows,
        "heatmap": heatmap_data,
        "timeseries": {
            "timestamp": df_down['timestamp'].tolist(),
            "velocity": df_down['velocity'].tolist()
        },
        "statistics": {
            "avg_velocity": avg_velocity,
            "fixation_ratio": fixation_ratio,
            "total_distance": total_distance,
            "most_viewed_area": most_viewed_area
        }
    }

@app.post("/predict")
async def predict(request: Request):
    raw = await request.json()

    try:
        window_features = extract_features_from_web(raw)

        # scaling
        data_baru_scaled = scaler.transform(window_features)

        # prediksi per window
        hasil = model.predict(data_baru_scaled)

        # probabilitas per window
        proba = model.predict_proba(data_baru_scaled)

        # majority voting
        hasil_akhir = Counter(hasil).most_common(1)[0][0]

        # confidence rata-rata
        mean_proba = np.mean(proba, axis=0)
        confidence = np.max(mean_proba) * 100

        # label text
        label_mapping = {
            0: 'normal',
            1: 'sedang',
            2: 'tinggi'
        }

        hasil_label = label_mapping[int(hasil_akhir)]

        return {
            'hasil_prediksi': int(hasil_akhir),
            'confidence': round(float(confidence), 2),
            'label': hasil_label,

            'detail_probabilitas': {
                'normal': round(float(mean_proba[0] * 100), 2),
                'sedang': round(float(mean_proba[1] * 100), 2),
                'tinggi': round(float(mean_proba[2] * 100), 2)
            },

            'total_window': len(hasil),

            'prediksi_per_window': hasil.tolist()
        }

    except Exception as e:
        print(f"Error: {e}")
