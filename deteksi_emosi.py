import os
import sys
import threading
import time
import traceback

import cv2
from deepface import DeepFace

# ================== PENGATURAN ==================
INDEKS_KAMERA = 0                                   # jalankan: python deteksi_emosi.py --cek
DETEKTOR = ["ssd", "opencv", "yunet", "retinaface"] # tombol 'd' untuk ganti
SISI_ANALISIS = 640                                 # sisi terpanjang frame saat dianalisis
UMUR_HASIL = 2.0

TERJEMAHAN = {"angry": "Marah", "disgust": "Jijik", "fear": "Takut", "happy": "Senang",
              "sad": "Sedih", "surprise": "Terkejut", "neutral": "Netral"}
WARNA = {"angry": (0, 0, 255), "disgust": (0, 140, 0), "fear": (200, 0, 200),
         "happy": (0, 220, 0), "sad": (255, 100, 0), "surprise": (0, 200, 255),
         "neutral": (200, 200, 200)}
ROTASI = [None, cv2.ROTATE_90_CLOCKWISE, cv2.ROTATE_180, cv2.ROTATE_90_COUNTERCLOCKWISE]
DERAJAT = [0, 90, 180, 270]

API = cv2.CAP_DSHOW if os.name == "nt" else cv2.CAP_ANY

# ================== DATA BERSAMA ==================
kunci = threading.Lock()
S = {"frame": None, "hasil": None, "waktu": 0.0, "status": "Menunggu frame...",
     "det": 0, "jalan": True}


def cek_kamera():
    """Daftar kamera 0-4: ukuran dan kecerahan, untuk mencari webcam Smart Connect."""
    print("Memeriksa kamera 0-4 ...")
    for i in range(5):
        cap = cv2.VideoCapture(i, API)
        if cap.isOpened():
            ok, f = False, None
            for _ in range(10):          # buang frame awal (sering hitam)
                ok, f = cap.read()
            if ok and f is not None:
                print(f"  indeks {i}: {f.shape[1]}x{f.shape[0]}, kecerahan rata-rata {f.mean():.0f}")
            else:
                print(f"  indeks {i}: terbuka, tapi tidak ada frame")
        cap.release()
    print("Pilih indeks yang kecerahannya normal (sekitar 60-180), lalu isi INDEKS_KAMERA.")


def buat_haar():
    """Haar hanya alat diagnosis; jika tidak tersedia, program tetap berjalan."""
    try:
        haar = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
        if haar.empty():
            return None
        return haar
    except Exception as e:
        print("Haar tidak tersedia (hanya untuk diagnosis):", e)
        return None


def thread_analisis():
    pesan_lalu = ""
    while S["jalan"]:
        with kunci:
            frame = None if S["frame"] is None else S["frame"].copy()
            det = DETEKTOR[S["det"]]
        if frame is None:
            time.sleep(0.05)
            continue

        h, w = frame.shape[:2]
        skala = min(1.0, SISI_ANALISIS / max(h, w))
        kecil = cv2.resize(frame, (0, 0), fx=skala, fy=skala) if skala < 1 else frame

        try:
            hasil = DeepFace.analyze(kecil, actions=["emotion"], detector_backend=det,
                                     enforce_detection=True, silent=True)
            if isinstance(hasil, list):
                hasil = hasil[0]
            r = hasil["region"]
            s = 1.0 / skala
            with kunci:
                S["hasil"] = {"emosi": hasil["dominant_emotion"], "skor": hasil["emotion"],
                              "kotak": (int(r["x"] * s), int(r["y"] * s),
                                        int(r["w"] * s), int(r["h"] * s))}
                S["waktu"] = time.time()
                S["status"] = f"OK ({det})"
        except ValueError as e:
            pesan = str(e)
            with kunci:
                S["hasil"] = None
                if "could not be detected" in pesan:
                    S["status"] = f"DeepFace: wajah tidak terdeteksi ({det})"
                else:
                    S["status"] = "ERROR: " + pesan[:70]
            if "could not be detected" not in pesan and pesan != pesan_lalu:
                print("\n[ERROR]", pesan)
                pesan_lalu = pesan
        except Exception as e:
            pesan = f"{type(e).__name__}: {e}"
            with kunci:
                S["hasil"] = None
                S["status"] = "ERROR: " + pesan[:70]
            if pesan != pesan_lalu:
                print("\n[ERROR]")
                traceback.print_exc()
                pesan_lalu = pesan
            time.sleep(1)


def gambar_label(frame, teks, x, y, warna):
    font = cv2.FONT_HERSHEY_SIMPLEX
    (tw, th), dasar = cv2.getTextSize(teks, font, 0.9, 2)
    yt = y - 10 if y - th - 14 > 0 else y + th + 20
    cv2.rectangle(frame, (x, yt - th - 8), (x + tw + 10, yt + dasar), warna, -1)
    cv2.putText(frame, teks, (x + 5, yt - 2), font, 0.9, (0, 0, 0), 2, cv2.LINE_AA)



def gambar_bar(frame, skor):
    # Panel background hitam solid
    panel_x, panel_y = 10, 10
    panel_w, panel_h = 285, 7 * 32 + 20

    cv2.rectangle(
        frame,
        (panel_x, panel_y),
        (panel_x + panel_w, panel_y + panel_h),
        (20, 20, 20),
        -1
    )

    # Border panel
    cv2.rectangle(
        frame,
        (panel_x, panel_y),
        (panel_x + panel_w, panel_y + panel_h),
        (100, 100, 100),
        1
    )

    y = panel_y + 25

    for emosi, nilai in sorted(skor.items(), key=lambda kv: -kv[1]):
        warna = WARNA.get(emosi, (255, 255, 255))
        nama = TERJEMAHAN.get(emosi, emosi)

        # Nama emosi
        cv2.putText(
            frame, nama, (panel_x + 10, y),
            cv2.FONT_HERSHEY_SIMPLEX, 0.48,
            (255, 255, 255), 1, cv2.LINE_AA
        )

        # Background bar
        bar_x = panel_x + 105
        bar_w = 105
        bar_h = 13
        bar_y = y - 11

        cv2.rectangle(
            frame,
            (bar_x, bar_y),
            (bar_x + bar_w, bar_y + bar_h),
            (65, 65, 65),
            -1
        )

        # Bar persentase
        panjang = int(bar_w * nilai / 100)

        if panjang > 0:
            cv2.rectangle(
                frame,
                (bar_x, bar_y),
                (bar_x + panjang, bar_y + bar_h),
                warna,
                -1
            )

        # Nilai persentase
        cv2.putText(
            frame, f"{nilai:.1f}%",
            (bar_x + bar_w + 8, y),
            cv2.FONT_HERSHEY_SIMPLEX, 0.45,
            (255, 255, 255), 1, cv2.LINE_AA
        )

        y += 32



def main():
    cap = cv2.VideoCapture(INDEKS_KAMERA, API)
    if not cap.isOpened():
        print("Kamera tidak bisa dibuka. Jalankan: python deteksi_emosi.py --cek")
        return
    for _ in range(15):                  # pemanasan: frame awal virtual camera sering hitam
        cap.read()

    haar = buat_haar()
    rot = 0
    threading.Thread(target=thread_analisis, daemon=True).start()
    print("Tombol: q=keluar | r=putar gambar | d=ganti detektor | s=simpan frame debug")

    while True:
        ok, frame = cap.read()
        if not ok:
            print("Gagal mengambil gambar dari kamera.")
            break

        if ROTASI[rot] is not None:
            frame = cv2.rotate(frame, ROTASI[rot])
        frame = cv2.flip(frame, 1)
        tinggi, lebar = frame.shape[:2]
        terang = frame.mean()

        with kunci:
            S["frame"] = frame.copy()
            hasil, umur = S["hasil"], time.time() - S["waktu"]
            status = S["status"]

        # Diagnosis independen: kotak BIRU = detektor Haar milik OpenCV
        if haar is not None:
            abu = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            k = 480 / max(tinggi, lebar)
            abu_kecil = cv2.resize(abu, (0, 0), fx=k, fy=k) if k < 1 else abu
            k = min(k, 1.0)
            for (x, y, w, h) in haar.detectMultiScale(abu_kecil, 1.1, 5, minSize=(40, 40)):
                x, y, w, h = int(x / k), int(y / k), int(w / k), int(h / k)
                cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 120, 0), 1)
                cv2.putText(frame, "Haar", (x, y + h + 18), cv2.FONT_HERSHEY_SIMPLEX,
                            0.5, (255, 120, 0), 1, cv2.LINE_AA)

        # Hasil emosi DeepFace: kotak berwarna
        if hasil is not None and umur < UMUR_HASIL:
            x, y, w, h = hasil["kotak"]
            warna = WARNA.get(hasil["emosi"], (0, 255, 0))
            label = f"{TERJEMAHAN.get(hasil['emosi'], hasil['emosi'])} {hasil['skor'][hasil['emosi']]:.0f}%"
            cv2.rectangle(frame, (x, y), (x + w, y + h), warna, 2)
            gambar_label(frame, label, x, y, warna)
            gambar_bar(frame, hasil["skor"])

        # Bar info diagnosis di bawah
        cv2.rectangle(frame, (0, tinggi - 28), (lebar, tinggi), (0, 0, 0), -1)
        info = f"{lebar}x{tinggi} | terang {terang:.0f} | putar {DERAJAT[rot]} | {status}"
        cv2.putText(frame, info, (8, tinggi - 9), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (0, 255, 255), 1, cv2.LINE_AA)

        cv2.imshow("Stand Pameran AI - Deteksi Emosi", frame)
        tombol = cv2.waitKey(1) & 0xFF
        if tombol == ord("q"):
            break
        elif tombol == ord("r"):
            rot = (rot + 1) % 4
        elif tombol == ord("d"):
            with kunci:
                S["det"] = (S["det"] + 1) % len(DETEKTOR)
                S["hasil"] = None
        elif tombol == ord("s"):
            with kunci:
                simpan = S["frame"]
            cv2.imwrite("debug_frame.jpg", simpan)
            print(f"Disimpan: debug_frame.jpg ({simpan.shape[1]}x{simpan.shape[0]})")

    S["jalan"] = False
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    if "--cek" in sys.argv:
        cek_kamera()
    else:
        main()