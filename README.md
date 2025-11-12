# 🧠 Deteksi Jenis Sampah Menggunakan CNN dan ML.NET

Proyek ini merupakan bagian dari tugas UTS mata kuliah **Machine Learning**, dengan tujuan membangun model **deteksi jenis sampah** menggunakan metode **Convolutional Neural Network (CNN)**.  
Model ini kemudian diintegrasikan ke dalam **framework ML.NET** untuk digunakan pada aplikasi berbasis .NET.

---

## 📌 Deskripsi Proyek
Aplikasi ini dibuat untuk membantu proses pengelolaan sampah dengan mengenali jenis sampah (organik, plastik, logam, dll) berdasarkan gambar.  
Proyek ini terdiri dari dua tahap utama:
1. **Pelatihan model CNN** menggunakan dataset gambar sampah.
2. **Implementasi model di ML.NET (C#)** untuk melakukan prediksi di lingkungan aplikasi.

---

## 📂 Dataset
Dataset yang digunakan adalah **Waste Classification Dataset** dari Kaggle, berisi beberapa kategori gambar sampah seperti:
- **Organic**
- **Plastic**
- **Metal**

Dataset diproses dengan ukuran gambar **224x224 piksel** dan normalisasi warna.

---

## 🧩 Arsitektur Model
Model dibangun menggunakan **Convolutional Neural Network (CNN)** dengan TensorFlow dan Keras.  
Proses pelatihan dilakukan selama **20 epoch**, dengan hasil akurasi akhir:

> 🎯 **Akurasi Model:** 46.88%  
> 📉 **Loss:** 1.2838

Model disimpan dalam format `.h5` dengan nama:
