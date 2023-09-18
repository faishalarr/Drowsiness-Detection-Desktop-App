import tkinter as tk
from tkinter import filedialog
import os
import shutil

def select_directory():
    directory_path = filedialog.askdirectory(title="Select a directory")
    if directory_path:
        move_files(directory_path)

def move_files(source_directory):
    target_directory = os.path.join("data", "images")
    os.makedirs(target_directory, exist_ok=True)
    
    for filename in os.listdir(source_directory):
        source_file = os.path.join(source_directory, filename)
        target_file = os.path.join(target_directory, filename)
        shutil.move(source_file, target_file)
    
    print("Files moved successfully!")

# Create the main window
root = tk.Tk()
root.title("File Mover")

# Create a button to select the directory
select_button = tk.Button(root, text="Select Directory", command=select_directory)
select_button.pack(padx=20, pady=20)

# Run the main loop
root.mainloop()
