import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import os

def open_directory():
    global dir_path
    dir_path = filedialog.askdirectory(initialdir=os.getcwd())
    if dir_path:
        global image_files
        image_files = [f for f in os.listdir(dir_path) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.gif'))]

        if image_files:
            global current_image_index
            current_image_index = 0
            show_image(os.path.join(dir_path, image_files[current_image_index]), current_image_index)

def show_image(image_path, image_index):
    image = Image.open(image_path)

    # Resize gambar agar tidak lebih besar dari 800x800 piksel
    max_size = (700, 700)
    image.thumbnail(max_size)

    photo = ImageTk.PhotoImage(image)
    image_label.config(image=photo)
    image_label.image = photo

    status_label.config(text=f"Image {image_index + 1} of {len(image_files)}")

    prev_button.config(state=tk.NORMAL if image_index > 0 else tk.DISABLED)
    next_button.config(state=tk.NORMAL if image_index < len(image_files) - 1 else tk.DISABLED)

def show_next_image():
    global current_image_index
    current_image_index += 1
    show_image(os.path.join(dir_path, image_files[current_image_index]), current_image_index)

def show_prev_image():
    global current_image_index
    current_image_index -= 1
    show_image(os.path.join(dir_path, image_files[current_image_index]), current_image_index)

app = tk.Tk()
app.title("Image Loader")
app.state('zoomed')

open_button = tk.Button(app, text="Open Directory", command=open_directory)
open_button.pack(pady=10)

prev_button = tk.Button(app, text="Previous", state=tk.DISABLED, command=show_prev_image)
prev_button.pack(side=tk.LEFT, padx=10)

next_button = tk.Button(app, text="Next", state=tk.DISABLED, command=show_next_image)
next_button.pack(side=tk.RIGHT, padx=10)

status_label = tk.Label(app, text="", pady=10)
status_label.pack()

image_label = tk.Label(app)
image_label.pack()

app.mainloop()
