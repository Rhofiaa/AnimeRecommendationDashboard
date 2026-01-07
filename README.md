# Anime Recommendation System

Panduan singkat memakai dashboard rekomendasi anime.

## Persiapan
- Masuk ke folder proyek di terminal.
- Instal semua paket: `pip install -r requirements.txt`.

## Menjalankan
- Jalankan aplikasi: `streamlit run app.py`.
- Jika muncul pesan Streamlit tidak dikenal atau tetap gagal, pakai perintah cadangan: `python -m streamlit run app.py`.
- Setelah berhasil, buka tautan yang muncul (biasanya http://localhost:8501).

## Dataset
- Pilih "Gunakan dataset bawaan" bila file `anime.csv` dan `rating.csv` sudah berada di folder proyek.
- Pilih "Upload manual" bila ingin mengunggah kedua file tersebut secara langsung lewat antarmuka.

## Catatan
- Atur jumlah tetangga dan jumlah rekomendasi lewat slider di sidebar sesuai kebutuhan.
