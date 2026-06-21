import streamlit as st
import pandas as pd
import joblib
import os

# Konfigurasi Halaman
st.set_page_config(page_title="Prediksi Churn Pelanggan", layout="wide")

st.title("Aplikasi Prediksi Churn Pelanggan")
st.write("Aplikasi ini dibuat untuk memprediksi apakah seorang pelanggan akan *churn* (berhenti berlangganan) atau *stay* (tetap bertahan) berdasarkan data historis penggunaan layanan.")

# Memuat model dan dependensinya
@st.cache_resource
def load_models():
    model = joblib.load('churn_model.joblib')
    scaler = joblib.load('scaler.joblib')
    features = joblib.load('model_features.joblib')
    return model, scaler, features

# Cek keberadaan file (Handling Error agar rapi)
if not os.path.exists('churn_model.joblib') or not os.path.exists('scaler.joblib') or not os.path.exists('model_features.joblib'):
    st.error("File model tidak ditemukan! Pastikan Anda sudah menjalankan script di Jupyter Notebook untuk menyimpan model.")
    st.info("Silakan cek instruksi di chat untuk mengetahui kode export model-nya.")
    st.stop()

model, scaler, expected_features = load_models()

st.write("---")
st.subheader("Masukkan Data Pelanggan Baru")

# Membagi form input menjadi 3 Tab agar tidak memanjang ke bawah
tab1, tab2, tab3 = st.tabs(["Demografi & Akun", "Aktivitas Penggunaan", "Transaksi & Kepuasan"])

with tab1:
    col1, col2 = st.columns(2)
    with col1:
        age = st.number_input("Umur (Age)", min_value=0, max_value=120, value=30)
        gender = st.selectbox("Jenis Kelamin (Gender)", ["Male", "Female", "Other"])
        country = st.text_input("Negara (Country)", value="Indonesia")
        city = st.text_input("Kota (City)", value="Jakarta")
    with col2:
        is_premium_user = st.radio("Pengguna Premium?", [1, 0], format_func=lambda x: "Ya" if x == 1 else "Tidak")
        subscription_type = st.selectbox("Tipe Langganan", ["Basic", "Standard", "Premium"])
        device_type = st.selectbox("Tipe Perangkat", ["Mobile", "Desktop", "Tablet"])
        acquisition_channel = st.selectbox("Jalur Akuisisi", ["Organic", "Paid Ads", "Referral", "Email Marketing"])

with tab2:
    col1, col2 = st.columns(2)
    with col1:
        total_visits = st.number_input("Total Kunjungan", min_value=0, value=10)
        avg_session_time = st.number_input("Rata-rata Waktu Sesi (menit)", min_value=0.0, value=5.0)
        pages_per_session = st.number_input("Halaman per Sesi", min_value=0.0, value=2.0)
    with col2:
        email_open_rate = st.number_input("Tingkat Buka Email", min_value=0.0, max_value=1.0, value=0.2)
        email_click_rate = st.number_input("Tingkat Klik Email", min_value=0.0, max_value=1.0, value=0.05)
        support_tickets = st.number_input("Jumlah Tiket Dukungan (Komplain)", min_value=0, value=0)

with tab3:
    col1, col2 = st.columns(2)
    with col1:
        total_spent = st.number_input("Total Pengeluaran", min_value=0.0, value=100.0)
        avg_order_value = st.number_input("Rata-rata Nilai Pesanan", min_value=0.0, value=20.0)
        lifetime_value = st.number_input("Nilai Seumur Hidup (Lifetime Value)", min_value=0.0, value=500.0)
        last_3_month_purchase_freq = st.number_input("Frekuensi Beli 3 Bulan Terakhir", min_value=0, value=2)
    with col2:
        payment_method = st.selectbox("Metode Pembayaran", ["Credit Card", "E-Wallet", "Bank Transfer"])
        discount_used = st.radio("Pernah Pakai Diskon?", [1, 0], format_func=lambda x: "Ya" if x == 1 else "Tidak")
        refund_requested = st.radio("Pernah Minta Refund?", [1, 0], format_func=lambda x: "Ya" if x == 1 else "Tidak")
        delivery_delay_days = st.number_input("Keterlambatan Pengiriman (Hari)", min_value=0, value=0)
        satisfaction_score = st.slider("Skor Kepuasan", min_value=1.0, max_value=5.0, value=4.0, step=0.1)
        nps_score = st.slider("Skor NPS", min_value=1.0, max_value=10.0, value=8.0, step=0.1)
        marketing_spend_per_user = st.number_input("Biaya Marketing per User", min_value=0.0, value=10.0)


st.write("---")
# Logika Prediksi
if st.button("Prediksi Status Churn", use_container_width=True):
    # 1. Menyusun data input menjadi DataFrame
    input_data = {
        'age': age,
        'gender': gender,
        'country': country,
        'city': city,
        'is_premium_user': is_premium_user,
        'subscription_type': subscription_type,
        'device_type': device_type,
        'acquisition_channel': acquisition_channel,
        'total_visits': total_visits,
        'avg_session_time': avg_session_time,
        'pages_per_session': pages_per_session,
        'email_open_rate': email_open_rate,
        'email_click_rate': email_click_rate,
        'support_tickets': support_tickets,
        'total_spent': total_spent,
        'avg_order_value': avg_order_value,
        'lifetime_value': lifetime_value,
        'last_3_month_purchase_freq': last_3_month_purchase_freq,
        'payment_method': payment_method,
        'discount_used': discount_used,
        'refund_requested': refund_requested,
        'delivery_delay_days': delivery_delay_days,
        'satisfaction_score': satisfaction_score,
        'nps_score': nps_score,
        'marketing_spend_per_user': marketing_spend_per_user
    }
    
    df_input = pd.DataFrame([input_data])
    
    # 2. Preprocessing: Membangun baris dummy secara manual sesuai `expected_features`
    # (pd.get_dummies dengan drop_first=True akan error/kosong jika diterapkan pada 1 baris)
    df_input_encoded = pd.DataFrame(0.0, index=[0], columns=expected_features)
    
    # Memasukkan nilai numerik
    kolom_numerik = df_input.select_dtypes(include='number').columns
    for col in kolom_numerik:
        if col in expected_features:
            df_input_encoded.at[0, col] = df_input.at[0, col]
            
    # Memasukkan nilai kategorikal (Manual One-Hot Encoding)
    kolom_kategori = df_input.select_dtypes(include='object').columns
    for col in kolom_kategori:
        val = df_input.at[0, col]
        dummy_col_name = f"{col}_{val}"
        # Jika kategori ini ada di data training, set nilainya jadi 1
        if dummy_col_name in expected_features:
            df_input_encoded.at[0, dummy_col_name] = 1
    
    # 4. Scaling (Standardization)
    df_input_scaled = scaler.transform(df_input_encoded)
    
    # 5. Memanggil Model untuk Prediksi
    prediction = model.predict(df_input_scaled)
    # Gunakan try-except karena SVM (jika bukan probability=True) mungkin tidak punya atribut predict_proba
    try:
        probability = model.predict_proba(df_input_scaled)[0]
        prob_churn = probability[1] * 100
        prob_stay = probability[0] * 100
    except:
        prob_churn = 100.0 if prediction[0] == 1 else 0.0
        prob_stay = 100.0 if prediction[0] == 0 else 0.0
    
    # 6. Menampilkan Hasil Visual
    st.subheader("Hasil Prediksi")
    if prediction[0] == 1:
        st.error(f"Pelanggan ini diprediksi **AKAN CHURN** (Berhenti berlangganan). Tingkat Keyakinan: {prob_churn:.1f}%")
        st.warning("**Rekomendasi Tindakan:** Segera hubungi pelanggan ini. Tawarkan promo retensi khusus, diskon perpanjangan, atau lakukan survei keluhan untuk mencegah mereka pergi.")
    else:
        st.success(f"Pelanggan ini diprediksi **AKAN STAY** (Tetap berlangganan). Tingkat Keyakinan: {prob_stay:.1f}%")
        st.info("**Rekomendasi Tindakan:** Jaga kualitas layanan. Pelanggan ini puas dengan layanan, tawarkan program *loyalty* atau fitur premium baru (Upselling/Cross-selling).")
