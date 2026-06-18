"""
train_model.py
----------------
Menjalankan ulang pipeline ML dari notebook (Kelompok_3_ML_Neural_Network.ipynb):
EDA -> preprocessing (LabelEncoder + StandardScaler) -> train/test split (stratify)
-> MLPClassifier (Neural Network) -> evaluasi.

Hasilnya disimpan ke disk (pickle + gambar PNG + metrics.json) supaya Flask app
(app.py) tidak perlu melatih ulang model setiap kali dashboard dibuka, dan supaya
fitur "coba prediksi sendiri" bisa pakai model yang sama tanpa training ulang.

Cara pakai:
    python train_model.py

Jalankan script ini SEKALI (atau setiap kali dataset berubah).
Setelah itu baru jalankan app.py.
"""

import json
import pickle

import matplotlib
matplotlib.use("Agg")  # biar bisa generate gambar tanpa GUI/display
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler

DATA_PATH = "data/ObesityDataSet_raw_and_data_sinthetic.xlsx"
MODEL_DIR = "model"
IMG_DIR = "static/img"

NUMERIC_COLS = ["Age", "Height", "Weight", "FCVC", "NCP", "CH2O", "FAF", "TUE"]


def main():
    # 1. Baca dataset ---------------------------------------------------
    df = pd.read_excel(DATA_PATH)

    dataset_info = {
        "n_rows": df.shape[0],
        "n_cols": df.shape[1],
        "columns": list(df.columns),
        "missing_values": int(df.isnull().sum().sum()),
    }

    # 2. EDA --------------------------------------------------------------
    plt.figure(figsize=(10, 5))
    sns.countplot(x="NObeyesdad", data=df, order=df["NObeyesdad"].value_counts().index)
    plt.xticks(rotation=45, ha="right")
    plt.title("Distribusi Tingkat Obesitas")
    plt.tight_layout()
    plt.savefig(f"{IMG_DIR}/distribusi.png", dpi=110)
    plt.close()

    plt.figure(figsize=(9, 7))
    sns.heatmap(df[NUMERIC_COLS].corr(), annot=True, cmap="coolwarm", fmt=".2f")
    plt.title("Heatmap Korelasi Variabel Numerik")
    plt.tight_layout()
    plt.savefig(f"{IMG_DIR}/korelasi.png", dpi=110)
    plt.close()

    # 3. Preprocessing -----------------------------------------------------
    df_processed = df.copy()
    label_encoders = {}
    categorical_cols = df_processed.select_dtypes(include=["object", "string"]).columns.tolist()
    # simpan urutan kategori ASLI (sebelum encoding) buat dropdown form prediksi
    categorical_options = {
        col: sorted(df[col].unique().tolist()) for col in categorical_cols if col != "NObeyesdad"
    }
    numeric_ranges = {
        col: [float(df[col].min()), float(df[col].max())] for col in NUMERIC_COLS
    }

    for col in categorical_cols:
        le = LabelEncoder()
        df_processed[col] = le.fit_transform(df_processed[col])
        label_encoders[col] = le

    X = df_processed.drop(columns=["NObeyesdad"])
    y = df_processed["NObeyesdad"]
    feature_columns = list(X.columns)

    # 4. Standardisasi ------------------------------------------------------
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # 5. Split data --------------------------------------------------------
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42, stratify=y
    )

    # 6. Training model (Neural Network) ------------------------------------
    model = MLPClassifier(
        hidden_layer_sizes=(100,),
        activation="relu",
        solver="adam",
        max_iter=1000,
        early_stopping=True,
        random_state=42,
    )
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # 7. Evaluasi -----------------------------------------------------------
    acc = accuracy_score(y_test, y_pred)
    class_names = list(label_encoders["NObeyesdad"].classes_)
    report_dict = classification_report(
        y_test, y_pred, target_names=class_names, output_dict=True, zero_division=0
    )
    cm = confusion_matrix(y_test, y_pred)

    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=class_names, yticklabels=class_names,
    )
    plt.xlabel("Prediksi")
    plt.ylabel("Aktual")
    plt.title("Confusion Matrix")
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(f"{IMG_DIR}/confusion_matrix.png", dpi=110)
    plt.close()

    # 8. Simpan model, encoder, scaler (pickle) ------------------------------
    with open(f"{MODEL_DIR}/model.pkl", "wb") as f:
        pickle.dump(model, f)
    with open(f"{MODEL_DIR}/label_encoders.pkl", "wb") as f:
        pickle.dump(label_encoders, f)
    with open(f"{MODEL_DIR}/scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)

    # 9. Simpan metrics & metadata buat dashboard + form prediksi -----------
    metrics = {
        "dataset_info": dataset_info,
        "accuracy": acc,
        "classification_report": report_dict,
        "class_names": class_names,
        "hidden_layer_sizes": list(model.hidden_layer_sizes) if isinstance(model.hidden_layer_sizes, tuple) else [model.hidden_layer_sizes],
        "n_iter": int(model.n_iter_),
        "n_layers": int(model.n_layers_),
        "train_size": X_train.shape[0],
        "test_size": X_test.shape[0],
        "n_features": X.shape[1],
        "feature_columns": feature_columns,
        "categorical_options": categorical_options,
        "numeric_ranges": numeric_ranges,
    }
    with open(f"{MODEL_DIR}/metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    print("Selesai melatih model (Neural Network / MLPClassifier).")
    print(f"Accuracy: {acc:.4f}")
    print(f"Jumlah iterasi training: {model.n_iter_}")
    print(f"Artefak tersimpan di '{MODEL_DIR}/' dan gambar di '{IMG_DIR}/'")


if __name__ == "__main__":
    main()
