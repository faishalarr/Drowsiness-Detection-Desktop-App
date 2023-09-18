import tkinter as tk
from tkinter import ttk
import pandas as pd

def show_column_data():
    # Replace 'your_file.csv' with the actual file path
    csv_file_path = 'yolov5/runs/train/exp/results.csv'
    
    # Load the CSV file into a pandas DataFrame
    df = pd.read_csv(csv_file_path)
    
    # Extract the 6th column using iloc
    column_data = df.iloc[:, 6]  # Adjust the column index (0-based) as needed
    
    # Calculate the mean of the column data
    column_mean = column_data.mean()
    
    # Clear any previous data in the treeviews
    for row in tree1.get_children():
        tree1.delete(row)
    for row in tree2.get_children():
        tree2.delete(row)
    
    # Insert the column data into the first treeview
    for value in column_data:
        tree1.insert('', 'end', values=(value,))
    
    # Insert the mean value into the second treeview
    tree2.insert('', 'end', values=(column_mean,))

# Create the main GUI window
root = tk.Tk()
root.title("CSV Column Viewer")

# Create the first treeview widget to display the original column data
tree1 = ttk.Treeview(root, columns=('Column6',), show='headings')
tree1.heading('#1', text='Column6')
tree1.pack()

# Create the second treeview widget to display the mean value
tree2 = ttk.Treeview(root, columns=('Mean',), show='headings')
tree2.heading('#1', text='Mean')
tree2.pack()

# Button to show column data
show_button = tk.Button(root, text="Show Column 6 Data and Mean", command=show_column_data)
show_button.pack()

# Run the GUI event loop
root.mainloop()
