import streamlit as st
import subprocess
import os
from PIL import Image

def run_tambahtesting():
    val_images_dir = os.path.join("dataset", "images", "val")

    # Create the directory if it doesn't exist
    os.makedirs(val_images_dir, exist_ok=True)

    # Handle file upload
    uploaded_files = st.file_uploader("Upload images", type=["jpg", "jpeg", "png"], accept_multiple_files=True)
    
    for uploaded_file in uploaded_files:
        file_contents = uploaded_file.read()

        # Determine the file path where you want to save the uploaded file
        save_path = os.path.join(val_images_dir, uploaded_file.name)

        with open(save_path, "wb") as f:
            f.write(file_contents)

    if uploaded_files:
        st.success("Files uploaded and saved successfully.")

def select_weights_directory():
    st.text("Running evaluation...")
    commandeval = f"python val.py --data custom_dataset.yaml --weights runs/train/exp/weights/best.pt"
    subprocess.run(commandeval, shell=True)
    st.text("Evaluation completed.")

def open_directory():
    dor_path = "runs/val/exp/"
    image_filea = [f for f in os.listdir(dor_path) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.gif'))]
    st.image(os.path.join(dor_path, image_filea[0]))
    # Add logic for iterating through images

def main():
    st.title("Model Evaluation")

    # Create tabs
    tabs = st.tabs(["Input Dataset", "Evaluate Model", "Show Results"])

    # Input Dataset tab
    with tabs[0]:
        st.text("This tab is for inputting the dataset.")
        run_tambahtesting()
    # Evaluate Model tab
    with tabs[1]:
        st.text("This tab is for evaluating the model.")
        if st.button("EVALUASI MODEL"):
            select_weights_directory()

    # Show Results tab
    with tabs[2]:
        st.text("This tab is for showing the evaluation results.")
        if st.button("TAMPILKAN HASIL"):
            open_directory()

if __name__ == "__main__":
    main()
