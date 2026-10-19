import os
import json
import base64
import cv2
import time
import csv
import pytesseract
import re
import numpy as np
import pandas as pd
from datetime import datetime
from deepface import DeepFace
from flask import Flask, render_template, request, redirect, url_for, session, flash

app = Flask(__name__)
app.secret_key = "secretkey123"
# Tambahkan baris ini untuk mengizinkan pengiriman data besar
app.config['MAX_CONTENT_LENGTH'] = 32 * 1024 * 1024  # 32 Megabytes

# Konfigurasi Folder
UPLOAD_FOLDER = "static/uploads"
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

def load_mahasiswa():
    with open("dataset/datamahasiswa2025.json") as file:
        data = json.load(file)
    return data["03.049"]

# =========================
# LOGIN
# =========================

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        npm_input = request.form["npm"]
        password_input = request.form["password"]
        mahasiswa = load_mahasiswa()

        for mhs in mahasiswa:
            if str(mhs["NIM"]) == npm_input and mhs["Password"] == password_input:
                session["user_npm"] = npm_input
                return redirect(url_for("scan_ktm"))
        
        return render_template("login.html", error="NPM atau Password salah")
    return render_template("login.html")

# =========================
# FASE 2 - OCR KTM & CROP WAJAH
# =========================

@app.route('/scan_ktm', methods=['GET', 'POST'])
def scan_ktm():
    if 'user_npm' not in session:
        return redirect(url_for('login'))

    if request.method == 'POST':
        ktm_b64 = request.form.get('ktm_base64')
        if not ktm_b64:
            flash("Gambar tidak diterima", "danger")
            return redirect(url_for('scan_ktm'))

        try:
            # Decode gambar
            img_data = base64.b64decode(ktm_b64.split(',')[1])
            user_npm = session['user_npm']
            
            # Simpan file KTM asli untuk arsip
            ktm_filepath = os.path.join(app.config['UPLOAD_FOLDER'], f"{user_npm}_ktm.jpg")
            with open(ktm_filepath, "wb") as fh:
                fh.write(img_data)

            # Load gambar untuk diproses
            img = cv2.imread(ktm_filepath)
            h_img, w_img, _ = img.shape

            # MANUAL CROP REFERENSI WAJAH ---
            # ambil area kiri (sesuai posisi kotak putih Area Wajah di UI)
            # Koordinat ini disesuaikan dengan posisi .guide-face di CSS (left: 8%, width: 28%)
            x_start = int(w_img * 0.05)
            x_end = int(w_img * 0.35)
            y_start = int(h_img * 0.15)
            y_end = int(h_img * 0.85)
            
            face_ref = img[y_start:y_end, x_start:x_end]
            ref_path = os.path.join(app.config['UPLOAD_FOLDER'], f"{user_npm}_ref.jpg")
            cv2.imwrite(ref_path, face_ref)

            # --- PROSES OCR NPM ---
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
            text = pytesseract.image_to_string(thresh, config='--psm 6')
            match = re.search(r'\b\d{10}\b', text)
            extracted_npm = match.group() if match else None

            # Validasi NPM
            if not extracted_npm or extracted_npm != user_npm:
                flash(f"NPM tidak cocok (Terdeteksi: {extracted_npm}). Pastikan NPM masuk kotak.", "danger")
                return redirect(url_for('scan_ktm'))

            session['ktm_verified'] = True
            session['reference_face'] = f"{user_npm}_ref.jpg"
            return redirect(url_for('verifikasi_wajah'))

        except Exception as e:
            flash(f"Error Sistem: {str(e)}", "danger")
            return redirect(url_for('scan_ktm'))

    return render_template('scan_ktm.html')

# =========================
# FASE 3 - HALAMAN FACE RECOGNITION
# =========================

@app.route("/verifikasi-wajah")
def verifikasi_wajah():
    if "ktm_verified" not in session:
        return redirect(url_for("scan_ktm"))
    return render_template("verify_face.html")

# Nama file log untuk hasil penelitian
LOG_PENELITIAN = "hasil_penelitian_face_rec.csv"

def simpan_log_csv(data):
    """Fungsi untuk mencatat hasil perbandingan ke file CSV secara otomatis"""
    file_exists = os.path.isfile(LOG_PENELITIAN)
    with open(LOG_PENELITIAN, mode='a', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=[
            "timestamp", "npm", "model", "verified", "distance", "threshold", "duration", "photo_path"
        ])
        if not file_exists:
            writer.writeheader()
        writer.writerow(data)

# @app.route("/api_verifikasi_wajah", methods=["POST"])
# def api_verifikasi_wajah():
#     try:
#         data = request.json.get("image")
#         user_npm = session.get('user_npm') # Ambil NPM dari session
        
#         if not data or not user_npm:
#             return {"status": "error", "message": "Data tidak lengkap"}

#         if "," in data:
#             data = data.split(",")[1]
#         img_data = base64.b64decode(data)
        
#         # Nama file: 2213020065_selfie_live.jpg (bisa ditambah timestamp jika ingin banyak percobaan)
#         timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
#         live_filename = f"{user_npm}_live_{timestamp_str}.jpg"
#         live_img_path = os.path.join(app.config['UPLOAD_FOLDER'], live_filename)
        
#         with open(live_img_path, "wb") as f:
#             f.write(img_data)
#             f.flush()
#             os.fsync(f.fileno())

#         ref_path = os.path.join(app.config['UPLOAD_FOLDER'], session.get('reference_face', ''))

#         if not os.path.exists(ref_path):
#             return {"status": "error", "message": "Foto referensi KTM tidak ditemukan. Silakan scan ulang KTM."}

#         #looping perbandingan Model (VGG-Face vs OpenFace)
#         models = ["VGG-Face", "OpenFace"]
#         result_summary = {}

#         for model_name in models:
#             start_time = time.time()

#             try:
#                 # Verifikasi menggunakan DeepFace
#                 result = DeepFace.verify(
#                     img1_path = ref_path, 
#                     img2_path = live_img_path,
#                     model_name = model_name,
#                     detector_backend = "opencv",
#                     enforce_detection = False # Agar tidak crash jika wajah kurang jelas
#                 )

#                 execution_time = round(time.time() - start_time, 4)

#                 #data untuk CSV
#                 log_data = {
#                     "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
#                     "npm": user_npm,
#                     "model": model_name,
#                     "verified": result["verified"],
#                     "distance": round(result["distance"], 4),
#                     "threshold": result["threshold"],
#                     "duration": execution_time,
#                     "photo_path": live_filename
#                 }

#                 #Simpan ke CSV secara otomatis
#                 simpan_log_csv(log_data)
#                 result_summary[model_name] = log_data

#             except Exception as e:
#                 result_summary[model_name] = {"error": str(e)}

#         is_verified = result_summary.get("VGG-Face", {}).get("verified", False)

#         if is_verified:
#             session["face_verified"] = True
#             return {"status": "success", "data": result_summary}
#         else:
#             return {"status": "fail", "message": "Wajah tidak cocok", "data": result_summary}

#     except Exception as e:
#         print(f"DEBUG ERROR FACE: {str(e)}")
#         return {"status": "error", "message": f"Gagal memproses verifikasi: {str(e)}"}


# versi kedua
# # Nama file log untuk hasil penelitian
# LOG_PENELITIAN = "hasil_penelitian_face_rec.csv"

# def simpan_log_csv(data):
#     """Fungsi untuk mencatat hasil perbandingan ke file CSV secara otomatis"""
#     file_exists = os.path.isfile(LOG_PENELITIAN)
#     with open(LOG_PENELITIAN, mode='a', newline='') as file:
#         writer = csv.DictWriter(file, fieldnames=[
#             "timestamp", "npm", "model", "verified", "distance", "threshold", "duration", "photo_path"
#         ])
#         if not file_exists:
#             writer.writeheader()
#         writer.writerow(data)

# @app.route("/api_verifikasi_wajah", methods=["POST"])
# def api_verifikasi_wajah():
#     try:
#         data = request.json.get("image")
#         user_npm = session.get('user_npm')
        
#         if not data or not user_npm:
#             return {"status": "error", "message": "Data tidak lengkap"}

#         # 1. Dekode dan Simpan Foto Live (Permanen sesuai NPM)
#         if "," in data:
#             data = data.split(",")[1]
#         img_data = base64.b64decode(data)
        
#         # Nama file: 2213020065_selfie_live.jpg (bisa ditambah timestamp jika ingin banyak percobaan)
#         timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
#         live_filename = f"{user_npm}_live_{timestamp_str}.jpg"
#         live_img_path = os.path.join(app.config['UPLOAD_FOLDER'], live_filename)
        
#         with open(live_img_path, "wb") as f:
#             f.write(img_data)

#         # 2. Ambil Path Referensi KTM
#         ref_path = os.path.join(app.config['UPLOAD_FOLDER'], session.get('reference_face', ''))
#         if not os.path.exists(ref_path):
#             return {"status": "error", "message": "Foto referensi KTM tidak ditemukan"}

#         # 3. Looping Perbandingan Model (VGG-Face vs OpenFace)
#         models = ["VGG-Face", "OpenFace"]
#         results_summary = {}

#         for model_name in models:
#             start_time = time.time()
            
#             try:
#                 # Proses Verifikasi
#                 result = DeepFace.verify(
#                     img1_path = ref_path, 
#                     img2_path = live_img_path,
#                     model_name = model_name,
#                     detector_backend = "opencv",
#                     enforce_detection = False
#                 )
                
#                 execution_time = round(time.time() - start_time, 4)
                
#                 # Data untuk CSV
#                 log_data = {
#                     "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
#                     "npm": user_npm,
#                     "model": model_name,
#                     "verified": result["verified"],
#                     "distance": round(result["distance"], 4),
#                     "threshold": result["threshold"],
#                     "duration": execution_time,
#                     "photo_path": live_filename
#                 }
                
#                 # Simpan ke CSV secara otomatis
#                 simpan_log_csv(log_data)
#                 results_summary[model_name] = log_data

#             except Exception as e:
#                 results_summary[model_name] = {"error": str(e)}

#         # 4. Logika Penentu Login (Misal: Harus VGG-Face yang True)
#         is_verified = results_summary.get("VGG-Face", {}).get("verified", False)

#         if is_verified:
#             session["face_verified"] = True
#             return {"status": "success", "data": results_summary}
#         else:
#             return {"status": "fail", "message": "Wajah tidak cocok", "data": results_summary}

#     except Exception as e:
#         print(f"DEBUG ERROR: {str(e)}")
#         return {"status": "error", "message": str(e)}

# versi ketiga
@app.route("/api_verifikasi_wajah", methods=["POST"])
def api_verifikasi_wajah():
    try:
        data = request.json.get("image")
        user_npm = session.get('user_npm')
        
        if not data or not user_npm:
            return {"status": "error", "message": "Data tidak lengkap"}

        if "," in data:
            data = data.split(",")[1]
        img_data = base64.b64decode(data)
        
        # --- PERBAIKAN 1: GUNAKAN PATH ABSOLUT ---
        timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{user_npm}_live_{timestamp_str}.jpg"
        # Path absolut agar stabil di Windows
        live_img_path = os.path.join(os.getcwd(), 'static', 'uploads', filename)
        
        with open(live_img_path, "wb") as f:
            f.write(img_data)
            f.flush()
            os.fsync(f.fileno())

        # Path Referensi Absolut
        ref_filename = session.get('reference_face', '')
        ref_path = os.path.join(os.getcwd(), 'static', 'uploads', ref_filename)

        if not os.path.exists(ref_path):
            return {"status": "error", "message": "Foto referensi KTM tidak ditemukan."}

        # --- PERBAIKAN 2: LOOPING & PENCATATAN CSV ---
        models = ["VGG-Face", "OpenFace"]
        final_summary = {}

        for model_name in models:
            start_time = time.time()
            try:
                result = DeepFace.verify(
                    img1_path = ref_path, 
                    img2_path = live_img_path,
                    model_name = model_name,
                    detector_backend = "opencv",
                    enforce_detection = False
                )
                
                execution_time = round(time.time() - start_time, 4)
                
                log_data = {
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "npm": user_npm,
                    "model": model_name,
                    "verified": result["verified"],
                    "distance": round(result["distance"], 4),
                    "threshold": result["threshold"],
                    "duration": execution_time,
                    "photo_path": filename
                }
                
                # Simpan ke CSV
                simpan_log_csv(log_data)
                final_summary[model_name] = log_data

            except Exception as e:
                # Tetap catat log meskipun error agar data penelitian tidak hilang
                error_log = {
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "npm": user_npm,
                    "model": model_name,
                    "verified": False,
                    "distance": 0,
                    "threshold": 0,
                    "duration": 0,
                    "photo_path": f"ERROR: {str(e)}"
                }
                simpan_log_csv(error_log)
                final_summary[model_name] = {"verified": False, "error": str(e)}

        # --- PERBAIKAN 3: PENGECEKAN HASIL LOGIN ---
        # Kita pakai VGG-Face sebagai penentu utama
        if final_summary.get("VGG-Face", {}).get("verified"):
            session["face_verified"] = True
            return {"status": "success", "data": final_summary}
        else:
            return {"status": "fail", "message": "Wajah tidak cocok", "data": final_summary}

    except Exception as e:
        return {"status": "error", "message": f"System Failure: {str(e)}"}

# =========================
# BILIK SUARA
# =========================

@app.route("/bilik-suara")
def bilik_suara():
    # Proteksi: Jika belum verifikasi wajah, tendang balik ke login
    if not session.get("face_verified"):
        flash("Selesaikan verifikasi wajah terlebih dahulu", "danger")
        return redirect(url_for("login"))
    
    mahasiswa = load_mahasiswa()
    npm = session.get("user_npm")

    nama = ""

    for mhs in mahasiswa:
        if str(mhs["NIM"]) == str(npm):
            nama = mhs["Nama"]
            break

    return render_template(
        "bilik_suara.html",
        nama_mahasiswa=nama
    )

@app.route("/submit_vote", methods=["POST"])
def submit_vote():
    if not session.get("face_verified"):
        return redirect(url_for("login"))
    
    pilihan = request.form.get("kandidat")
    npm = session.get("user_npm")
    
    # Simpan ke file suara.json atau database
    # ... logika simpan suara ...

    session.clear() # Hapus session agar tidak bisa vote lagi
    return "Terima kasih, suara Anda telah direkam!"

@app.route("/panel-riset") # URL Tersembunyi
def admin_research():
    if not os.path.exists(LOG_PENELITIAN):
        return "Belum ada data penelitian."
    
    df = pd.read_csv(LOG_PENELITIAN)

    total_uji = len(df) // 2
    
    # --- ANALISIS DATA UNTUK SKRIPSI ---
    # 1. Hitung berapa kali setiap NPM melakukan uji coba
    # Karena 1x klik simpan 2 baris (VGG & OpenFace), kita bagi 2
    rekap_npm = df.groupby('npm').size() // 2
    rekap_npm = rekap_npm.to_dict()

    # 2. Ambil statistik rata-rata distance untuk melihat akurasi
    # Semakin kecil distance, semakin akurat model tersebut
    stats_distance = df.groupby('model')['distance'].mean().to_dict()

    # 3. Data untuk Grafik Kecepatan
    stats_durasi = df[df['duration'] > 0].groupby('model')['duration'].mean().to_dict()
    chart_data = {
        "labels": ["VGG-Face", "OpenFace"],
        "durasi": [round(stats_durasi.get("VGG-Face", 0), 4), round(stats_durasi.get("OpenFace", 0), 4)],
        "distance": [round(stats_distance.get("VGG-Face", 0), 4), round(stats_distance.get("OpenFace", 0), 4)]
    }

    # Ambil data mahasiswa untuk mapping nama
    list_mahasiswa = load_mahasiswa()
    mapping_nama = {str(mhs["NIM"]): mhs["Nama"] for mhs in list_mahasiswa}

    # 4. Data Tabel (Semua history uji coba)
    reports = df.sort_values(by="timestamp", ascending=False).to_dict(orient="records")
    
    # Tambahkan Nama ke dalam setiap baris laporan
    for row in reports:
        row['nama_mhs'] = mapping_nama.get(str(row['npm']), "Nama Tidak Ditemukan")
        
    return render_template(
        "admin_research.html", 
        reports=reports, 
        rekap_npm=rekap_npm, 
        chart_data=chart_data,
        total_uji=total_uji)

if __name__ == "__main__":
    app.run(debug=True) 