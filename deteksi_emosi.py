import math
import os
import sys
import threading
import time
import traceback
from collections import deque

import cv2
import numpy as np
from deepface import DeepFace

# ================== PENGATURAN ==================
INDEKS_KAMERA = 0                                   # jalankan: python deteksi_emosi.py --cek
DETEKTOR = ["ssd", "opencv", "yunet", "retinaface"] # tombol 'd' untuk ganti
SISI_ANALISIS = 640                                 # sisi terpanjang frame saat mencari wajah
UMUR_HASIL = 1.5
NAMA_JENDELA = "Python OpenCV-Emotion Detection"
LEBAR_PANEL = 300       # lebar panel indikator di sisi kiri (px, pada resolusi kamera)
FOLDER_STIKER = os.path.join(os.path.dirname(os.path.abspath(__file__)), "stickers")

# --- Pengaturan akurasi emosi (boleh disetel) ---
MARGIN_EMOSI = (0.02, 0.12)  # variasi margin potongan wajah; hasilnya dirata-ratakan
PAKAI_CLAHE = True      # normalisasi kontras/pencahayaan pada wajah
PAKAI_FLIP = True       # rata-ratakan hasil gambar asli + cerminnya (emosi dan umur)
PAKAI_LURUS = True      # putar wajah agar garis mata mendatar (butuh posisi mata dari detektor)
PANGKAT_TAJAM = 1.3     # >1 mempertegas emosi dominan & mengecilkan campuran (1.0 = mati)
ALPHA_SKOR = 0.35       # penghalusan skor (kecil = lebih stabil, besar = lebih responsif)
ALPHA_KOTAK = 0.5       # penghalusan posisi kotak wajah
HISTERESIS = 8.0        # selisih skor (poin %) agar label emosi boleh berganti
LUPA_WAJAH = 1.0        # detik tanpa wajah sebelum penghalusan/kalibrasi di-reset
UKURAN_MIN_WAJAH = 48   # piksel; wajah lebih kecil dari ini diabaikan
DURASI_KALIBRASI = 2.5  # detik pengambilan wajah netral saat menekan 'c'
# Koreksi bias model. 1.0 = tanpa koreksi. Ubah pelan-pelan sesuai hasil uji di lokasi.
BOBOT = {"angry": 0.95, "disgust": 0.85, "fear": 0.95, "happy": 1.05,
         "sad": 0.80, "surprise": 1.10, "neutral": 0.90}

# --- Deteksi umur ---
PAKAI_UMUR = True       # False = matikan deteksi umur
INTERVAL_UMUR = 1.0     # detik antar-estimasi umur (model umur lebih berat)
RIWAYAT_UMUR = 15       # jumlah hasil terakhir yang dirata-ratakan (terpangkas)
MIN_RIWAYAT_UMUR = 3    # umur baru ditampilkan setelah sekian hasil terkumpul
MARGIN_UMUR = 0.25      # margin potongan wajah untuk model umur
UKURAN_MIN_UMUR = 80    # piksel; wajah lebih kecil tidak dipakai untuk umur

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
     "det": 0, "jalan": True, "kal_minta": False, "kal_hapus": False}


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


# ================== PEMROSESAN WAJAH ==================
def potong_wajah(frame, x, y, w, h, margin):
    """Potong wajah dari frame resolusi penuh, dengan tambahan margin di sekelilingnya."""
    tinggi, lebar = frame.shape[:2]
    dx, dy = int(w * margin), int(h * margin)
    x1, y1 = max(0, x - dx), max(0, y - dy)
    x2, y2 = min(lebar, x + w + dx), min(tinggi, y + h + dy)
    return frame[y1:y2, x1:x2]


def luruskan_wajah(frame, area, s):
    """Putar frame agar garis mata mendatar (kepala miring jadi tegak).
    Kembalikan (frame_untuk_analisis, (x, y, w, h)) pada koordinat frame tersebut.
    Jika detektor tidak memberi posisi mata, frame dikembalikan apa adanya."""
    x, y = int(area["x"] * s), int(area["y"] * s)
    w, h = int(area["w"] * s), int(area["h"] * s)
    asli = (frame, (x, y, w, h))
    if not PAKAI_LURUS:
        return asli
    mata_a, mata_b = area.get("left_eye"), area.get("right_eye")
    if not mata_a or not mata_b:
        return asli
    try:
        (x1, y1), (x2, y2) = sorted([tuple(mata_a), tuple(mata_b)])   # urut dari kiri ke kanan
        x1, y1, x2, y2 = x1 * s, y1 * s, x2 * s, y2 * s
        sudut = math.degrees(math.atan2(y2 - y1, x2 - x1))
    except Exception:
        return asli
    if abs(sudut) < 2 or abs(sudut) > 35:
        return asli
    M = cv2.getRotationMatrix2D(((x1 + x2) / 2, (y1 + y2) / 2), sudut, 1.0)
    tinggi, lebar = frame.shape[:2]
    putar = cv2.warpAffine(frame, M, (lebar, tinggi), flags=cv2.INTER_LINEAR,
                           borderMode=cv2.BORDER_REPLICATE)
    cx, cy = x + w / 2, y + h / 2
    nx = M[0, 0] * cx + M[0, 1] * cy + M[0, 2]
    ny = M[1, 0] * cx + M[1, 1] * cy + M[1, 2]
    return putar, (int(nx - w / 2), int(ny - h / 2), w, h)


def perbaiki_kontras(wajah, clahe):
    """CLAHE pada kanal terang (LAB) supaya cahaya miring/redup tidak merusak fitur wajah."""
    lab = cv2.cvtColor(wajah, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    l = clahe.apply(l)
    return cv2.cvtColor(cv2.merge((l, a, b)), cv2.COLOR_LAB2BGR)


def skor_emosi(wajah):
    """Skor emosi untuk satu potongan wajah (tanpa deteksi ulang)."""
    hasil = DeepFace.analyze(wajah, actions=["emotion"], detector_backend="skip",
                             enforce_detection=False, silent=True)
    if isinstance(hasil, list):
        hasil = hasil[0]
    return {k: float(v) for k, v in hasil["emotion"].items()}


def skor_emosi_ensemble(dasar, x, y, w, h, clahe):
    """Rata-rata skor dari beberapa variasi potongan (margin berbeda) + versi cerminnya."""
    daftar = []
    for margin in MARGIN_EMOSI:
        crop = potong_wajah(dasar, x, y, w, h, margin)
        if crop.size == 0:
            continue
        if PAKAI_CLAHE:
            crop = perbaiki_kontras(crop, clahe)
        daftar.append(skor_emosi(crop))
        if PAKAI_FLIP:
            daftar.append(skor_emosi(cv2.flip(crop, 1)))
    if not daftar:
        raise ValueError("Face could not be detected (crop kosong)")
    return rata_rata(daftar)


def estimasi_umur(wajah):
    """Perkiraan umur dari potongan wajah (tanpa deteksi ulang)."""
    hasil = DeepFace.analyze(wajah, actions=["age"], detector_backend="skip",
                             enforce_detection=False, silent=True)
    if isinstance(hasil, list):
        hasil = hasil[0]
    return float(hasil["age"])


def umur_ensemble(dasar, x, y, w, h):
    """Umur dari potongan dengan margin lebih longgar (sesuai data latih model umur) + cermin."""
    crop = potong_wajah(dasar, x, y, w, h, MARGIN_UMUR)
    if crop.size == 0:
        return None
    nilai = [estimasi_umur(crop)]
    if PAKAI_FLIP:
        nilai.append(estimasi_umur(cv2.flip(crop, 1)))
    return sum(nilai) / len(nilai)


def umur_terpangkas(riwayat):
    """Rata-rata setelah membuang 20% nilai terendah dan tertinggi (tahan terhadap outlier)."""
    urut = sorted(riwayat)
    p = int(len(urut) * 0.2)
    inti = urut[p:len(urut) - p] or urut
    return sum(inti) / len(inti)


def rata_rata(daftar):
    return {k: sum(d[k] for d in daftar) / len(daftar) for k in daftar[0]}


def normalisasi(skor):
    total = sum(skor.values()) or 1.0
    return {k: v * 100.0 / total for k, v in skor.items()}


def terapkan_bobot(skor):
    return normalisasi({k: v * BOBOT.get(k, 1.0) for k, v in skor.items()})


def pertajam(skor, pangkat):
    """Pangkat > 1 memperbesar jarak emosi dominan dengan emosi lain (campuran kecil menyusut)."""
    if pangkat == 1.0:
        return skor
    return normalisasi({k: max(v, 1e-6) ** pangkat for k, v in skor.items()})


def terapkan_baseline(skor, dasar):
    """Kurangi skor 'wajah netral' orang ini: emosi non-netral hanya dihitung bila melebihi baseline-nya."""
    hasil = {}
    for k, v in skor.items():
        hasil[k] = v if k == "neutral" else max(0.0, v - dasar.get(k, 0.0))
    return normalisasi(hasil)


def haluskan(lama, baru, alpha):
    if lama is None:
        return dict(baru) if isinstance(baru, dict) else tuple(baru)
    if isinstance(baru, dict):
        return {k: alpha * baru[k] + (1 - alpha) * lama[k] for k in baru}
    return tuple(alpha * n + (1 - alpha) * l for n, l in zip(baru, lama))


def thread_analisis():
    pesan_lalu = ""
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))
    skor_halus, kotak_halus, emosi_tampil = None, None, None
    riwayat_umur, waktu_umur, umur_aktif = deque(maxlen=RIWAYAT_UMUR), 0.0, PAKAI_UMUR
    baseline, kal_sampel, kal_mulai = None, None, 0.0
    terakhir_wajah = time.time()

    while S["jalan"]:
        with kunci:
            frame = None if S["frame"] is None else S["frame"].copy()
            det = DETEKTOR[S["det"]]
            minta_kal, hapus_kal = S["kal_minta"], S["kal_hapus"]
            S["kal_minta"], S["kal_hapus"] = False, False
        if hapus_kal:
            baseline, kal_sampel, skor_halus, emosi_tampil = None, None, None, None
        if minta_kal:
            kal_sampel, kal_mulai = [], time.time()
        if frame is None:
            time.sleep(0.05)
            continue

        h, w = frame.shape[:2]
        skala = min(1.0, SISI_ANALISIS / max(h, w))
        kecil = cv2.resize(frame, (0, 0), fx=skala, fy=skala) if skala < 1 else frame
        s = 1.0 / skala

        try:
            # 1) Cari wajah pada frame kecil (cepat), ambil yang terbesar
            wajah_list = DeepFace.extract_faces(kecil, detector_backend=det,
                                                enforce_detection=True, align=False)
            terbesar = max(wajah_list,
                           key=lambda f: f["facial_area"]["w"] * f["facial_area"]["h"])
            a = terbesar["facial_area"]
            kx, ky, kw, kh = int(a["x"] * s), int(a["y"] * s), int(a["w"] * s), int(a["h"] * s)
            if min(kw, kh) < UKURAN_MIN_WAJAH:
                raise ValueError("Face could not be detected (terlalu kecil)")

            # 2) Luruskan wajah (garis mata mendatar) pada frame resolusi penuh
            dasar, (ax, ay, aw, ah) = luruskan_wajah(frame, a, s)

            # 3) Emosi: banyak variasi potongan -> rata-rata -> bobot -> pertajam
            mentah = pertajam(terapkan_bobot(skor_emosi_ensemble(dasar, ax, ay, aw, ah, clahe)),
                              PANGKAT_TAJAM)

            # 4) Kalibrasi wajah netral (tombol 'c'), bila diminta / aktif
            if kal_sampel is not None:
                kal_sampel.append(mentah)
                if time.time() - kal_mulai >= DURASI_KALIBRASI and len(kal_sampel) >= 3:
                    baseline, kal_sampel = rata_rata(kal_sampel), None
                    skor_halus, emosi_tampil = None, None
            skor_pakai = terapkan_baseline(mentah, baseline) if baseline else mentah

            # 5) Penghalusan antar-frame + histeresis label
            skor_halus = haluskan(skor_halus, skor_pakai, ALPHA_SKOR)
            kotak_halus = haluskan(kotak_halus, (kx, ky, kw, kh), ALPHA_KOTAK)
            terbaik = max(skor_halus, key=skor_halus.get)
            if emosi_tampil is None or terbaik == emosi_tampil:
                emosi_tampil = terbaik
            elif skor_halus[terbaik] - skor_halus[emosi_tampil] > HISTERESIS:
                emosi_tampil = terbaik

            # 6) Umur: dihitung berkala, rata-rata terpangkas dari beberapa hasil terakhir
            if umur_aktif and time.time() - waktu_umur >= INTERVAL_UMUR and min(aw, ah) >= UKURAN_MIN_UMUR:
                try:
                    u = umur_ensemble(dasar, ax, ay, aw, ah)
                    if u is not None:
                        riwayat_umur.append(u)
                except Exception as e:
                    umur_aktif = False
                    print("\n[UMUR] Gagal, deteksi umur dimatikan:", e)
                waktu_umur = time.time()
            if not umur_aktif:
                umur_tampil = -1                # -1 = tidak tersedia
            elif len(riwayat_umur) >= MIN_RIWAYAT_UMUR:
                umur_tampil = int(round(umur_terpangkas(riwayat_umur)))
            else:
                umur_tampil = None              # belum cukup data

            ket = " | KALIBRASI..." if kal_sampel is not None else (" | kalibrasi" if baseline else "")
            terakhir_wajah = time.time()
            with kunci:
                S["hasil"] = {"emosi": emosi_tampil, "skor": dict(skor_halus),
                              "kotak": tuple(int(v) for v in kotak_halus),
                              "umur": umur_tampil}
                S["waktu"] = time.time()
                S["status"] = f"OK ({det}){ket}"
        except ValueError as e:
            pesan = str(e)
            if time.time() - terakhir_wajah > LUPA_WAJAH:
                skor_halus, kotak_halus, emosi_tampil = None, None, None
                riwayat_umur.clear()
                waktu_umur = 0.0
                baseline, kal_sampel = None, None   # kemungkinan orang berganti
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


# ================== TAMPILAN ==================
def gambar_label(frame, teks, x, y, warna):
    font = cv2.FONT_HERSHEY_SIMPLEX
    (tw, th), dasar = cv2.getTextSize(teks, font, 0.9, 2)
    yt = y - 10 if y - th - 14 > 0 else y + th + 20
    cv2.rectangle(frame, (x, yt - th - 8), (x + tw + 10, yt + dasar), warna, -1)
    cv2.putText(frame, teks, (x + 5, yt - 2), font, 0.9, (0, 0, 0), 2, cv2.LINE_AA)


STIKER = {}            # emosi (inggris) -> gambar stiker, diisi di main()
_CACHE_STIKER = {}


def muat_stiker():
    """Baca stiker dari folder 'stickers' (nama file = nama emosi Indonesia: senang.jpg, dst)."""
    ada, hilang = {}, []
    for kunci_emosi, nama in TERJEMAHAN.items():
        img = None
        for ext in (".png", ".jpg", ".jpeg", ".webp"):
            path = os.path.join(FOLDER_STIKER, nama.lower() + ext)
            if os.path.isfile(path):
                img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
                if img is not None:
                    break
        if img is None:
            hilang.append(nama.lower())
            continue
        if img.ndim == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        ada[kunci_emosi] = img
    if hilang:
        print("Stiker tidak ditemukan untuk:", ", ".join(hilang), "| folder:", FOLDER_STIKER)
    return ada


def siapkan_stiker(emosi, lebar, tinggi):
    """Stiker diperkecil agar muat (rasio tetap); PNG transparan dicampur dengan latar panel."""
    kunci_cache = (emosi, lebar, tinggi)
    if kunci_cache in _CACHE_STIKER:
        return _CACHE_STIKER[kunci_cache]
    img = STIKER.get(emosi)
    if img is None:
        return None
    h, w = img.shape[:2]
    k = min(lebar / w, tinggi / h)
    nw, nh = max(1, int(w * k)), max(1, int(h * k))
    kecil = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA if k < 1 else cv2.INTER_LINEAR)
    if kecil.shape[2] == 4:
        alfa = kecil[:, :, 3:4].astype(np.float32) / 255.0
        bgr = kecil[:, :, :3].astype(np.float32)
        kecil = (bgr * alfa + 20 * (1 - alfa)).astype(np.uint8)
    _CACHE_STIKER[kunci_cache] = kecil
    return kecil


def buat_panel(tinggi, skor, umur=None, emosi=None):
    """Panel di sisi kiri video: skor emosi, umur, lalu stiker sesuai ekspresi."""
    font = cv2.FONT_HERSHEY_SIMPLEX
    panel = np.full((tinggi, LEBAR_PANEL, 3), 20, np.uint8)
    cv2.rectangle(panel, (0, 0), (LEBAR_PANEL - 1, tinggi - 1), (100, 100, 100), 1)
    cv2.putText(panel, "SKOR EMOSI", (12, 22), font, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

    if skor is None:
        cv2.putText(panel, "Menunggu wajah...", (12, 50), font, 0.5,
                    (160, 160, 160), 1, cv2.LINE_AA)
        return panel

    # --- bar skor (ringkas) ---
    y = 46
    for nama_emosi, nilai in sorted(skor.items(), key=lambda kv: -kv[1]):
        warna = WARNA.get(nama_emosi, (255, 255, 255))
        nama = TERJEMAHAN.get(nama_emosi, nama_emosi)
        cv2.putText(panel, nama, (12, y), font, 0.42, (255, 255, 255), 1, cv2.LINE_AA)

        bar_x, bar_w, bar_h = 95, 120, 11
        bar_y = y - 10
        cv2.rectangle(panel, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (65, 65, 65), -1)
        panjang = int(bar_w * nilai / 100)
        if panjang > 0:
            cv2.rectangle(panel, (bar_x, bar_y), (bar_x + panjang, bar_y + bar_h), warna, -1)
        cv2.putText(panel, f"{nilai:.1f}%", (bar_x + bar_w + 6, y), font, 0.4,
                    (255, 255, 255), 1, cv2.LINE_AA)
        y += 22

    # --- umur (satu baris) ---
    y0 = y - 4
    cv2.line(panel, (12, y0), (LEBAR_PANEL - 12, y0), (100, 100, 100), 1)
    yu = y0 + 26
    cv2.putText(panel, "Umur", (12, yu), font, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
    if umur is None:
        cv2.putText(panel, "Menghitung...", (95, yu), font, 0.5, (160, 160, 160), 1, cv2.LINE_AA)
    elif umur < 0:
        cv2.putText(panel, "Tidak tersedia", (95, yu), font, 0.5, (160, 160, 160), 1, cv2.LINE_AA)
    else:
        cv2.putText(panel, f"~{umur} th", (95, yu), font, 0.8, (0, 220, 255), 2, cv2.LINE_AA)

    # --- stiker di bawahnya, mengisi sisa ruang ---
    atas, bawah = yu + 14, tinggi - 10
    area_h, area_w = bawah - atas, LEBAR_PANEL - 24
    if emosi is not None and area_h >= 60:
        st = siapkan_stiker(emosi, area_w, area_h)
        if st is not None:
            sh, sw = st.shape[:2]
            x = (LEBAR_PANEL - sw) // 2
            yy = atas + (area_h - sh) // 2
            panel[yy:yy + sh, x:x + sw] = st
    return panel


def atur_fullscreen(aktif):
    if aktif:
        cv2.setWindowProperty(NAMA_JENDELA, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    else:
        cv2.setWindowProperty(NAMA_JENDELA, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(NAMA_JENDELA, 1280, 720)


def sesuaikan_ke_jendela(frame):
    """Skala frame ke ukuran jendela tanpa mengubah rasio (sisa ruang diisi hitam)."""
    try:
        _, _, ww, wh = cv2.getWindowImageRect(NAMA_JENDELA)
    except cv2.error:
        return frame
    if ww <= 0 or wh <= 0:
        return frame
    h, w = frame.shape[:2]
    k = min(ww / w, wh / h)
    nw, nh = int(w * k), int(h * k)
    if nw <= 0 or nh <= 0:
        return frame
    interp = cv2.INTER_LINEAR if k > 1 else cv2.INTER_AREA
    kecil = cv2.resize(frame, (nw, nh), interpolation=interp)
    atas, kiri = (wh - nh) // 2, (ww - nw) // 2
    return cv2.copyMakeBorder(kecil, atas, wh - nh - atas, kiri, ww - nw - kiri,
                              cv2.BORDER_CONSTANT, value=(0, 0, 0))


def main():
    cap = cv2.VideoCapture(INDEKS_KAMERA, API)
    if not cap.isOpened():
        print("Kamera tidak bisa dibuka. Jalankan: python deteksi_emosi.py --cek")
        return
    for _ in range(15):                  # pemanasan: frame awal virtual camera sering hitam
        cap.read()

    cv2.namedWindow(NAMA_JENDELA, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(NAMA_JENDELA, 1280, 720)
    layar_penuh = "--full" in sys.argv
    if layar_penuh:
        atur_fullscreen(True)

    haar = buat_haar()
    STIKER.update(muat_stiker())
    rot = 0
    threading.Thread(target=thread_analisis, daemon=True).start()
    print("Tombol: q=keluar | f=fullscreen | Esc=keluar dari fullscreen | "
          "r=putar gambar | d=ganti detektor | s=simpan frame debug | "
          "c=kalibrasi wajah netral | x=hapus kalibrasi")

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
            u = hasil.get("umur")
            if u is not None and u >= 0:
                label += f" | {u} th"
            cv2.rectangle(frame, (x, y), (x + w, y + h), warna, 2)
            gambar_label(frame, label, x, y, warna)

        # Bar info diagnosis di bawah
        cv2.rectangle(frame, (0, tinggi - 28), (lebar, tinggi), (0, 0, 0), -1)
        info = f"{lebar}x{tinggi} | terang {terang:.0f} | putar {DERAJAT[rot]} | {status}"
        cv2.putText(frame, info, (8, tinggi - 9), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (0, 255, 255), 1, cv2.LINE_AA)

        ada_hasil = hasil is not None and umur < UMUR_HASIL
        panel = buat_panel(tinggi, hasil["skor"] if ada_hasil else None,
                           hasil.get("umur") if ada_hasil else None,
                           hasil["emosi"] if ada_hasil else None)
        tampil = cv2.hconcat([panel, frame])
        cv2.imshow(NAMA_JENDELA, sesuaikan_ke_jendela(tampil))
        tombol = cv2.waitKey(1) & 0xFF
        if tombol == ord("q"):
            break
        elif tombol == ord("f"):
            layar_penuh = not layar_penuh
            atur_fullscreen(layar_penuh)
        elif tombol == 27 and layar_penuh:      # Esc
            layar_penuh = False
            atur_fullscreen(False)
        elif tombol == ord("c"):
            with kunci:
                S["kal_minta"] = True
            print("Kalibrasi: tahan wajah NETRAL menghadap kamera selama beberapa detik ...")
        elif tombol == ord("x"):
            with kunci:
                S["kal_hapus"] = True
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

        # Jendela ditutup lewat tombol X
        if cv2.getWindowProperty(NAMA_JENDELA, cv2.WND_PROP_VISIBLE) < 1:
            break

    S["jalan"] = False
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    if "--cek" in sys.argv:
        cek_kamera()
    else:
        main()