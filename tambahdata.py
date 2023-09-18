import tkinter as tk
from tkinter import filedialog
import shutil
import os

def select_files(destination_folder):
    file_paths = filedialog.askopenfilenames(filetypes=[("Image files", "*.jpg *.jpeg *.png")])

    for file_path in file_paths:
        file_name = os.path.basename(file_path)
        destination_path = os.path.join(destination_folder, file_name)
        shutil.copy(file_path, destination_path)
        print(f"Added {file_name} to {destination_folder}")

root = tk.Tk()
root.withdraw()  # Hide the main window

default_destination_train = 'dataset/images/train/'
select_files(default_destination_train)

default_destination_val = 'dataset/images/val'
select_files(default_destination_val)
