import os
import random
import shutil
import tkinter as tk
from tkinter import filedialog

# Inisialisasi pop-up window untuk memilih direktori
root = tk.Tk()
root.withdraw()  # Sembunyikan main window

# Memilih direktori sumber
source_dir = filedialog.askdirectory(title="Pilih direktori sumber")

# Memilih direktori tujuan
destination_base_dir = "./dataset/"

# Tentukan rasio pembagian
split_ratio = 0.9

# Buat subdirektori untuk set pelatihan dan validasi
train_image_dir = os.path.join(destination_base_dir, "images/train")
val_image_dir = os.path.join(destination_base_dir, "images/val")
train_label_dir = os.path.join(destination_base_dir, "labels/train")
val_label_dir = os.path.join(destination_base_dir, "labels/val")

os.makedirs(train_image_dir, exist_ok=True)
os.makedirs(val_image_dir, exist_ok=True)
os.makedirs(train_label_dir, exist_ok=True)
os.makedirs(val_label_dir, exist_ok=True)

# Dapatkan daftar file gambar dan label
image_files = [filename for filename in os.listdir(source_dir) if filename.endswith(".jpg")]
label_files = [filename for filename in os.listdir(source_dir) if filename.endswith(".txt")]

# Acak urutan daftar untuk memastikan keacakan
random.shuffle(image_files)

# Hitung indeks pembagian
split_index = int(len(image_files) * split_ratio)

# Bagi daftar menjadi set pelatihan dan validasi
train_images = image_files[:split_index]
val_images = image_files[split_index:]

# Pindahkan file gambar dan file label yang sesuai ke direktori yang tepat
for image_filename in train_images:
    image_source_path = os.path.join(source_dir, image_filename)
    label_filename = os.path.splitext(image_filename)[0] + ".txt"
    label_source_path = os.path.join(source_dir, label_filename)
    shutil.copy(image_source_path, os.path.join(train_image_dir, image_filename))
    shutil.copy(label_source_path, os.path.join(train_label_dir, label_filename))

for image_filename in val_images:
    image_source_path = os.path.join(source_dir, image_filename)
    label_filename = os.path.splitext(image_filename)[0] + ".txt"
    label_source_path = os.path.join(source_dir, label_filename)
    shutil.copy(image_source_path, os.path.join(val_image_dir, image_filename))
    shutil.copy(label_source_path, os.path.join(val_label_dir, label_filename))

print("Pembagian dataset selesai.")
