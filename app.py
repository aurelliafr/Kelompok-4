"""
app.py
-------
Flask dashboard yang nampilin output dari pipeline ML
Kelompok_3_ML_Neural_Network.ipynb (EDA, evaluasi model MLPClassifier /
Neural Network: akurasi, classification report, confusion matrix) DAN
form interaktif buat coba prediksi sendiri pakai model yang sudah dilatih.

PENTING: app ini cuma BACA model & metrics yang sudah disimpan oleh
train_model.py (model/model.pkl, model/label_encoders.pkl,
model/scaler.pkl, model/metrics.json). Jadi tidak ada training ulang
setiap kali halaman dibuka / setiap kali ada yang submit form prediksi.

Cara pakai:
    1. python train_model.py      (sekali aja, atau tiap dataset ganti)
    2. python app.py
    3. buka http://127.0.0.1:5000
"""

import json
import pickle

import numpy as np
import pandas as pd
from flask import Flask, render_template, request

app = Flask(__name__)

MODEL_DIR = "model"

# ---- load semua artefak SEKALI saat app start, bukan setiap request ----
with open(f"{MODEL_DIR}/model.pkl", "rb") as f:
    model = pickle.load(f)
with open(f"{MODEL_DIR}/label_encoders.pkl", "rb") as f:
    label_encoders = pickle.load(f)
with open(f"{MODEL_DIR}/scaler.pkl", "rb") as f:
    scaler = pickle.load(f)
with open(f"{MODEL_DIR}/metrics.json", "r") as f:
    metrics = json.load(f)

FEATURE_COLUMNS = metrics["feature_columns"]
CATEGORICAL_OPTIONS = metrics["categorical_options"]
NUMERIC_RANGES = metrics["numeric_ranges"]
CLASS_NAMES = metrics["class_names"]

# Label & deskripsi singkat tiap fitur, buat ditampilkan di form (Bahasa Indonesia)
FIELD_INFO = {
    "Gender": {"label": "Jenis kelamin", "type": "select"},
    "Age": {"label": "Usia (tahun)", "type": "number"},
    "Height": {"label": "Tinggi badan (meter)", "type": "number"},
    "Weight": {"label": "Berat badan (kg)", "type": "number"},
    "family_history_with_overweight": {"label": "Riwayat keluarga obesitas", "type": "select"},
    "FAVC": {"label": "Sering makan tinggi kalori", "type": "select"},
    "FCVC": {"label": "Frekuensi makan sayur (1-3)", "type": "number"},
    "NCP": {"label": "Jumlah makan besar / hari (1-4)", "type": "number"},
    "CAEC": {"label": "Makan di antara waktu makan", "type": "select"},
    "SMOKE": {"label": "Merokok", "type": "select"},
    "CH2O": {"label": "Konsumsi air / hari, liter (1-3)", "type": "number"},
    "SCC": {"label": "Memantau kalori yang dikonsumsi", "type": "select"},
    "FAF": {"label": "Frekuensi aktivitas fisik (0-3)", "type": "number"},
    "TUE": {"label": "Waktu pakai gadget, jam (0-2)", "type": "number"},
    "CALC": {"label": "Konsumsi alkohol", "type": "select"},
    "MTRANS": {"label": "Transportasi yang digunakan", "type": "select"},
}


def build_form_fields(submitted=None):
    """Siapin definisi field form prediksi, lengkap dengan value default /
    value yang baru disubmit user (biar form nggak balik kosong)."""
    fields = []
    for col in FEATURE_COLUMNS:
        info = FIELD_INFO[col]
        field = {"name": col, "label": info["label"], "type": info["type"]}

        if info["type"] == "select":
            field["options"] = CATEGORICAL_OPTIONS[col]
            field["value"] = (submitted or {}).get(col, field["options"][0])
        else:
            lo, hi = NUMERIC_RANGES[col]
            field["min"] = lo
            field["max"] = hi
            default_val = round((lo + hi) / 2, 1)
            field["value"] = (submitted or {}).get(col, default_val)

        fields.append(field)
    return fields


def build_dashboard_context():
    report = metrics["classification_report"]
    class_names = metrics["class_names"]

    report_rows = []
    for cls in class_names:
        row = report[cls]
        report_rows.append(
            {
                "label": cls,
                "precision": row["precision"],
                "recall": row["recall"],
                "f1_score": row["f1-score"],
                "support": int(row["support"]),
            }
        )

    summary_rows = [
        {
            "label": "Macro avg",
            "precision": report["macro avg"]["precision"],
            "recall": report["macro avg"]["recall"],
            "f1_score": report["macro avg"]["f1-score"],
            "support": int(report["macro avg"]["support"]),
        },
        {
            "label": "Weighted avg",
            "precision": report["weighted avg"]["precision"],
            "recall": report["weighted avg"]["recall"],
            "f1_score": report["weighted avg"]["f1-score"],
            "support": int(report["weighted avg"]["support"]),
        },
    ]

    return {
        "dataset_info": metrics["dataset_info"],
        "accuracy": metrics["accuracy"],
        "report_rows": report_rows,
        "summary_rows": summary_rows,
        "hidden_layer_sizes": metrics["hidden_layer_sizes"],
        "n_iter": metrics["n_iter"],
        "n_layers": metrics["n_layers"],
        "train_size": metrics["train_size"],
        "test_size": metrics["test_size"],
        "n_features": metrics["n_features"],
    }


@app.route("/")
def dashboard():
    context = build_dashboard_context()
    context["form_fields"] = build_form_fields()
    context["prediction"] = None
    return render_template("dashboard.html", **context)


@app.route("/predict", methods=["POST"])
def predict():
    submitted = request.form.to_dict()

    # susun satu baris fitur sesuai urutan kolom training
    row = []
    for col in FEATURE_COLUMNS:
        raw_value = submitted.get(col, "")
        if col in label_encoders:  # kolom kategorikal -> encode jadi angka
            encoded = label_encoders[col].transform([raw_value])[0]
            row.append(encoded)
        else:  # kolom numerik
            row.append(float(raw_value))

    X_input = pd.DataFrame([row], columns=FEATURE_COLUMNS)
    X_scaled = scaler.transform(X_input)

    pred_encoded = model.predict(X_scaled)[0]
    pred_label = label_encoders["NObeyesdad"].inverse_transform([pred_encoded])[0]

    probabilities = model.predict_proba(X_scaled)[0]
    confidence = float(np.max(probabilities)) * 100

    # urutan kelas pada predict_proba ikut model.classes_ (= label encoded)
    prob_rows = sorted(
        [
            {
                "label": label_encoders["NObeyesdad"].inverse_transform([cls])[0],
                "prob": float(p) * 100,
            }
            for cls, p in zip(model.classes_, probabilities)
        ],
        key=lambda r: r["prob"],
        reverse=True,
    )

    context = build_dashboard_context()
    context["form_fields"] = build_form_fields(submitted=submitted)
    context["prediction"] = {
        "label": pred_label,
        "confidence": confidence,
        "prob_rows": prob_rows,
    }
    return render_template("dashboard.html", **context)


if __name__ == "__main__":
    app.run(debug=True)
