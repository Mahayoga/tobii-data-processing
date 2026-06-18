import json
import os
import numpy as np
import pandas as pd
import mysql.connector

# ==========================================
# CONFIG
# ==========================================

DATASET_FOLDER = "datasets/new_datasets"

SCREEN_WIDTH = 1920
SCREEN_HEIGHT = 1080

GRID_SIZE = 3

FIXATION_THRESHOLD = 0.02
WINDOW_SECONDS = 10
SAMPLING_RATE = 3

WINDOW_SIZE = WINDOW_SECONDS * SAMPLING_RATE

db = mysql.connector.connect(
  host="localhost",
  user="root",
  password="root",
  database="smart_anxiety_06_14_2026"
)

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

def extract_features(file_path):

    with open(file_path, "r") as f:
        data = json.load(f)

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

        features = {

            "avg_velocity":
                window_df['velocity'].mean(),

            "max_velocity":
                window_df['velocity'].max(),

            "std_velocity":
                window_df['velocity'].std(),

            "total_distance":
                window_df['distance'].sum(),

            "fixation_ratio":
                window_df['fixation'].mean(),

            "unique_area_count":
                window_df['area'].nunique()
        }

        all_window_features.append(features)

    return all_window_features

# ==========================================
# PROCESS ALL FILES
# ==========================================

all_features = []
data_count = 1

for file_name in os.listdir(DATASET_FOLDER):

    if file_name.endswith(".json"):

        file_path = os.path.join(
            DATASET_FOLDER,
            file_name
        )

        try:

            window_features = extract_features(file_path)

            cursor_db = db.cursor()
            sql = "SELECT skor_kecemasan FROM log_hasil_kecemasan WHERE id_test_sessions = %s"
            cursor_db.execute(sql, (file_name.split("_")[2].split(".")[0],))
            result = cursor_db.fetchone()

            for features in window_features:

                features['responden'] = f"responden_{data_count}"

                if result is not None:
                    skor_kecemasan = result[0]

                    if skor_kecemasan <= 37:
                        features['label'] = "normal"
                    elif skor_kecemasan <= 44:
                        features['label'] = "sedang"
                    elif skor_kecemasan <= 80:
                        features['label'] = "tinggi"
                    else:
                        features['label'] = "tidak valid"

                all_features.append(features)

            # responden count
            features['responden'] = f"responden_{data_count}"

            # ==================================
            # LABEL MANUAL
            # ==================================
            # nanti ganti dari database STAI-T

            # features['label'] = "tinggi"

            all_features.append(features)

            print(f"Processed: {file_name}")
            data_count += 1

        except Exception as e:
            print(f"Error {file_name}: {e}")

# ==========================================
# SAVE CSV
# ==========================================

dataset_df = pd.DataFrame(all_features)

dataset_df.to_csv(
    "eye_tracking_dataset_27_responden.csv",
    index=False
)

print("\nDataset saved!")
print(dataset_df.head())