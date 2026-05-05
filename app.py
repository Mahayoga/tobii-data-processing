from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import numpy as np

app = FastAPI()

# Biar bisa diakses dari frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/extract")
async def extract_features(request: Request):
    raw = await request.json()

    # handle format fleksibel
    if isinstance(raw, dict):
        raw = raw.get("data", [])

    df = pd.DataFrame(raw)

    # =============================
    # 🔧 PREPROCESSING
    # =============================
    df = df.sort_values(by="timestamp").reset_index(drop=True)

    # normalisasi (optional, sesuaikan resolusi)
    SCREEN_W, SCREEN_H = 1920, 1080
    df['x'] = df['gaze_x'] / SCREEN_W
    df['y'] = df['gaze_y'] / SCREEN_H

    df['dx'] = df['x'].diff()
    df['dy'] = df['y'].diff()
    df['dt'] = df['timestamp'].diff() / 1000

    df = df.fillna(0)

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
        "timeseries": {
            "timestamp": df_down['timestamp'].tolist(),
            "velocity": df_down['velocity'].tolist()
        }
    }