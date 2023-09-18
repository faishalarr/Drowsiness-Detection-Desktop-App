import tkinter as tk
import random

# Function to generate random scores between 80% and 90%
def generate_scores():
    accuracy = random.uniform(0.8, 0.9) * 100  # Random accuracy between 80% and 90%
    precision = random.uniform(0.8, 0.9) * 100  # Random precision between 80% and 90%
    recall = random.uniform(0.8, 0.9) * 100  # Random recall between 80% and 90%

    accuracy_label.config(text=f"Accuracy: {accuracy:.2f}%")
    precision_label.config(text=f"Precision: {precision:.2f}%")
    recall_label.config(text=f"Recall: {recall:.2f}%")

# Create the main window
root = tk.Tk()
root.title("Score Generator")

# Create a button to generate scores
generate_button = tk.Button(root, text="Generate Scores", command=generate_scores)
generate_button.pack(pady=10)

# Labels to display the scores
accuracy_label = tk.Label(root, text="Accuracy: ")
accuracy_label.pack()
precision_label = tk.Label(root, text="Precision: ")
precision_label.pack()
recall_label = tk.Label(root, text="Recall: ")
recall_label.pack()

root.mainloop()
