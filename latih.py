import subprocess
import tkinter as tk
from tkinter import Label, Entry, Button

def latihdata(batch_size, num_epochs):
    command = f"python train.py --img 640 --batch {batch_size} --epochs {num_epochs} --data custom_dataset.yaml --weights yolov5s.pt --cache"
    subprocess.run(f'start /wait cmd /k "{command}"', shell=True)

def start_training():
    batch_size = int(batch_entry.get())
    num_epochs = int(epoch_entry.get())
    latihdata(batch_size, num_epochs)

# Create a GUI window
root = tk.Tk()
root.title("Training Configuration")

# Batch size input
batch_label = Label(root, text="Batch Size:")
batch_label.pack()
batch_entry = Entry(root)
batch_entry.pack()

# Number of epochs input
epoch_label = Label(root, text="Number of Epochs:")
epoch_label.pack()
epoch_entry = Entry(root)
epoch_entry.pack()

# Start training button
start_button = Button(root, text="Start Training", command=start_training)
start_button.pack()

root.mainloop()
