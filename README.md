# Dashboard Klasifikasi Tingkat Obesitas (Flask + Neural Network)

Dashboard ini menampilkan ulang hasil dari notebook
`Kelompok_3_ML_Neural_Network.ipynb`: EDA (distribusi & korelasi), info
dataset, evaluasi model Neural Network / MLPClassifier (akurasi,
classification report, confusion matrix), **dan form interaktif untuk
mencoba prediksi sendiri** memakai model yang sudah dilatih.

## Struktur folder

```
project/
├── app.py                  # Flask app: dashboard + route /predict
├── train_model.py          # Latih model dari xlsx, simpan model + gambar
├── requirements.txt
├── data/
│   └── ObesityDataSet_raw_and_data_sinthetic.xlsx
├── model/                  # dibuat otomatis oleh train_model.py
│   ├── model.pkl            # model MLPClassifier terlatih
│   ├── label_encoders.pkl   # encoder per kolom kategorikal + target
│   ├── scaler.pkl            # StandardScaler yang dipakai saat training
│   └── metrics.json          # akurasi, classification report, dst.
├── static/
│   ├── style.css
│   └── img/                # distribusi.png, korelasi.png, confusion_matrix.png
└── templates/
    └── dashboard.html
```

## Cara jalanin

1. (Opsional) buat virtual environment:
   ```
   python -m venv venv
   venv\Scripts\activate        # Mac/Linux: source venv/bin/activate
   ```

2. Install dependency:
   ```
   pip install -r requirements.txt
   ```

3. **Latih model & generate semua gambar** (sekali saja, atau ulangi kalau
   dataset di `data/` diganti):
   ```
   python train_model.py
   ```
   Kalau berhasil, muncul `Selesai melatih model (Neural Network / MLPClassifier).`

4. **Jalankan dashboard:**
   ```
   python app.py
   ```
   Buka browser ke `http://127.0.0.1:5000`.

## Fitur "Coba Sendiri" (prediksi interaktif)

Di bagian bawah dashboard ada form berisi semua 16 fitur yang dipakai model
(jenis kelamin, usia, tinggi, berat, kebiasaan makan, dst). Saat tombol
**"Prediksi Sekarang"** diklik:

1. Browser kirim data form ke route `/predict` (di `app.py`) pakai HTTP POST.
2. Flask susun data itu jadi satu baris fitur sesuai urutan kolom training,
   encode kolom kategorikal pakai `label_encoders.pkl`, lalu standardisasi
   pakai `scaler.pkl` — **persis** preprocessing yang sama dengan saat training.
3. Baris itu dimasukkan ke `model.pkl` (model MLP yang sudah dilatih) untuk
   diprediksi, lalu hasilnya (kategori + tingkat keyakinan per kelas)
   ditampilkan kembali di halaman yang sama.

Tidak ada training ulang yang terjadi saat prediksi — model cuma dipakai
untuk `.predict()`, jadi responsnya instan.

## Kenapa dipisah jadi 2 file (train_model.py & app.py)?

Supaya dashboard (dan fitur prediksi) tidak perlu melatih ulang model setiap
kali halaman dibuka / setiap kali ada yang submit form. `train_model.py`
melatih model sekali, simpan hasilnya (`model.pkl`, `label_encoders.pkl`,
`scaler.pkl`, `metrics.json`, gambar EDA & confusion matrix). `app.py`
tinggal **membaca** semua itu sekali saat start, lalu pakai berkali-kali.

## Mengganti dataset

Ganti file `data/ObesityDataSet_raw_and_data_sinthetic.xlsx` dengan data
baru (format kolom harus sama: 16 fitur + kolom target `NObeyesdad`), lalu
jalankan ulang `python train_model.py`. Dashboard otomatis menampilkan hasil
yang baru setelah `app.py` di-restart.
