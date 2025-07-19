
import streamlit as st
import cv2
import numpy as np
from datetime import datetime
import requests
import os

# Konfigurasi
WEBHOOK_URL = "https://script.google.com/macros/s/AKfycbwkR5y8iS6aSHBvtfknR1RDdkb3b1VGt-7LZW5unlIKpMENqXlh7kSv_lxpzFNEnjgBZg/exec"
MODEL_PATH = "opencv_trained_model.yml"
CASCADE_PATH = "haarcascade_frontalface_default.xml"

# Muat model dan classifier
face_cascade = cv2.CascadeClassifier(CASCADE_PATH)
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read(MODEL_PATH)

# Pemetaan label ID ke nama
label_dict = {}
dataset_path = "dataset"
for idx, nama in enumerate(os.listdir(dataset_path)):
    if os.path.isdir(os.path.join(dataset_path, nama)):
        label_dict[idx] = nama

st.set_page_config(page_title="Absensi Wajah", layout="centered")
st.title("📸 Sistem Absensi Wajah")

start = st.button("🔴 Mulai Absensi")
frame_placeholder = st.empty()

absen_status = {}

def get_status_absen():
    now = datetime.now().time()
    if datetime.strptime("06:00", "%H:%M").time() <= now <= datetime.strptime("07:30", "%H:%M").time():
        return "Hadir"
    elif now <= datetime.strptime("14:55", "%H:%M").time():
        return "Terlambat"
    elif now <= datetime.strptime("17:00", "%H:%M").time():
        return "Pulang"
    return None

if start:
    cap = cv2.VideoCapture(0)
    st.info("Tekan tombol Stop di atas untuk menghentikan.")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            st.error("❌ Kamera gagal.")
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            wajah = gray[y:y+h, x:x+w]
            label, confidence = recognizer.predict(wajah)

            if confidence < 80:
                nama = label_dict.get(label, "Tidak dikenal")
                status = get_status_absen()
                jam = datetime.now().strftime("%H:%M:%S")

                if not status:
                    st.warning(f"{nama} datang di luar jam absensi.")
                    continue

                if status in ["Hadir", "Terlambat"]:
                    if nama in absen_status:
                        continue
                    absen_status[nama] = status
                    try:
                        r = requests.post(WEBHOOK_URL, data={"nama": nama, "waktu": jam, "status": status})
                        if r.status_code == 200:
                            st.success(f"[{nama}] sudah absen ({status}) jam {jam}")
                    except:
                        st.error("❌ Gagal kirim absen ke Google Sheets.")
                elif status == "Pulang":
                    if absen_status.get(nama) == "Pulang":
                        continue
                    absen_status[nama] = "Pulang"
                    try:
                        r = requests.post(WEBHOOK_URL, data={"nama": nama, "waktu": jam, "status": status})
                        if r.status_code == 200:
                            st.success(f"[{nama}] sudah absen Pulang jam {jam}")
                    except:
                        st.error("❌ Gagal kirim absen pulang.")

                cv2.putText(frame, f"{nama} ({status})", (x, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_placeholder.image(frame_rgb, channels="RGB")

    cap.release()
    st.success("✅ Kamera dimatikan.")
