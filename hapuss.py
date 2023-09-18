import os
import tkinter as tk
from tkinter import messagebox, filedialog

def clear_directory(directory_path):
    try:
        for filename in os.listdir(directory_path):
            file_path = os.path.join(directory_path, filename)
            if os.path.isfile(file_path):
                os.remove(file_path)
            elif os.path.isdir(file_path):
                clear_directory(file_path)
                os.rmdir(file_path)
        return True
    except Exception as e:
        return str(e)

def clear_datrain_directory():
    directory_path = "data/datrain"
    result = clear_directory(directory_path)
    if result is True:
        messagebox.showinfo("Sukses", f"Isi dari direktori {directory_path} berhasil dihapus.")
    else:
        messagebox.showerror("Kesalahan", f"Terjadi kesalahan: {result}")

# Membuat GUI menggunakan Tkinter
root = tk.Tk()
root.title("Hapus Isi Direktori datrain")

label = tk.Label(root, text="Klik tombol untuk menghapus isi direktori data/datrain:")
label.pack(pady=10)

clear_button = tk.Button(root, text="Hapus Isi datrain", command=clear_datrain_directory)
clear_button.pack()

root.mainloop()
