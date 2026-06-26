# Prediksi Churn Pelanggan - Data Science Capstone

Proyek ini adalah tugas besar (Capstone Project) Data Science dari Bengkel Koding, yang berfokus pada analisis perilaku pelanggan dan pembuatan model Machine Learning untuk memprediksi **Customer Churn** (potensi pelanggan berhenti berlangganan).

## Tujuan Proyek
Memprediksi apakah seorang pelanggan akan **Stay (0)** atau **Churn (1)** berdasarkan data demografis, perilaku transaksi, dan interaksi marketing. Proyek ini dilengkapi dengan *Dashboard* interaktif berbasis **Streamlit** untuk mensimulasikan hasil prediksi secara *real-time*.

## Struktur Repositori
- `UAS_bengkod.ipynb` : *Jupyter Notebook* utama yang berisi alur kerja end-to-end Data Science (Mulai dari EDA, Preprocessing, SMOTE, Hyperparameter Tuning, hingga Ekspor Model).
- `app.py` : Skrip antarmuka web interaktif menggunakan *Streamlit* untuk deployment model.
- `Sales - Marketing customer dataset.csv` : Dataset mentah yang berisi data profil dan histori transaksi dari 15.000 pelanggan.
- `churn_model.joblib` : File model Machine Learning (Tuned) terbaik hasil pelatihan.
- `scaler.joblib` : File *StandardScaler* untuk menormalisasi data *input* pengguna baru.
- `model_features.joblib` : File berisi urutan nama kolom *one-hot encoding* agar sesuai dengan blueprint awal.

## Alur Kerja (Pipeline)
Proyek ini mengikuti panduan terstruktur 5 Modul:
1. **Exploratory Data Analysis (EDA):** Deteksi outlier, identifikasi *missing values*, dan visualisasi *imbalanced target* (Keseimbangan Kelas Churn).
2. **Modeling Baseline:** Uji coba performa model kotor (tanpa preprocessing) untuk tolok ukur.
3. **Data Preprocessing:** Imputasi median/modus, penghapusan fitur *leakage* & duplikat, dan *One-Hot Encoding*.
4. **Validasi & Sampling:** Pembagian data Latih-Uji (80:20), *Scaling*, dan menyeimbangkan kelas target dengan teknik **SMOTE**.
5. **Hyperparameter Tuning:** Mencari parameter terbaik menggunakan *GridSearchCV* untuk *Logistic Regression, Random Forest*, dan *Voting Classifier (SVM + KNN)*.

## Cara Menjalankan Aplikasi Web (Streamlit)

1. Pastikan Anda telah menginstal pustaka yang dibutuhkan:
   ```bash
   pip install pandas numpy scikit-learn streamlit imbalanced-learn joblib
   ```
2. Jalankan aplikasi Streamlit melalui terminal:
   ```bash
   streamlit run app.py
   ```
3. Buka browser pada alamat `http://localhost:8501`. 
4. Masukkan parameter profil pelanggan pada menu di sebelah kiri untuk melihat hasil prediksi (Churn/Stay) beserta rekomendasinya!

---
*Dibuat untuk memenuhi tugas akhir Data Science.*
