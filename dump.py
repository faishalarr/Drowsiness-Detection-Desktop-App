from tkinter import *
from PIL import ImageTk, Image
import cv2
import numpy as np
import argparse
import time
import pygame
import tkinter as tk
from tkinter import filedialog
from datetime import date, datetime
import subprocess
import shutil
import os
import sys
from plyer import notification
import customtkinter
import PySimpleGUI as sg



try:
    subprocess.check_call(["pip", "install", "-r", "requirements.txt"])
except subprocess.CalledProcessError:
    print("An error occurred while installing required packages.")


root = customtkinter.CTk()
frame = customtkinter.CTkFrame(master=root)
frame.pack(pady=10, padx=10, fill="both", expand=True)
customtkinter.set_appearance_mode("light")
customtkinter.set_default_color_theme("green")

root.title("Deteksi Kantuk")
root.iconbitmap("D:/ical/Magang Astra/Object-Detection-using-Yolov5-main/Object Detection using Yolov5/yolov5/runs/train/exp2/weights/Daihatsu.ico")
root.geometry('1920x1080')
root.state('zoomed')


#gambar icon 1
img1 = Image.open('kantuk.jpg')
cam = img1.resize((300,300),Image.ANTIALIAS)
cam1 = ImageTk.PhotoImage(cam)

deskripsi = customtkinter.CTkLabel(master=root, 
text="Kantuk (Drowsiness) adalah kondisi ketika seseorang atau individu membutuhkan tidur. \nRasa kantuk membuat seseorang menjadi kurang memperhatikan lingkungan \nsekitar dan lebih mudah terganggu.",
fg_color="transparent",
font=("Arial", 20))
deskripsi.place(relx=0.5,anchor="n",y=380)

#gambar icon 2
img2 = Image.open('D:/ical/Magang Astra/SPEK/loupe.png')
det = img2.resize((50,50),Image.ANTIALIAS)
det1 = ImageTk.PhotoImage(det)

title = customtkinter.CTkLabel(master=root, text="Deteksi Kantuk",fg_color="transparent",font=("Arial", 28))
title.place(relx=0.5,anchor="n",y=40)

cuy2 = Label(image=cam1,fg='#000000')
cuy2.pack()
cuy2.place(relx=0.5,anchor="n",y=150)

#Button Detect
def run_detect():
    root.withdraw()  # Hide the root window temporarily

    initial_dir = r"D:/ical/Magang Astra/Object Detection using Yolov5/yolov5/runs/train"

    
    model_path = filedialog.askopenfilename(title="Select ONNX Model File", initialdir=initial_dir, filetypes=[("ONNX Files", "*.onnx")])

    if not model_path:
        print("No model selected. Exiting...")
        return
    model_directory = os.path.dirname(model_path)
    img_w = 640
    img_h = 640
    classes_file = 'classes.txt'

    def class_name():
        classes=[]
        file= open(classes_file,'r')
        while True:
            name=file.readline().strip('\n')
            classes.append(name)
            if not name:
                break
        return classes
        
    def detection1(img, net, classes): 
        blob = cv2.dnn.blobFromImage(img, 1/255 , (img_w, img_h), swapRB=True, mean=(0,0,0), crop= False)
        
        net.setInput(blob)
        t1= time.time()
        outputs= net.forward(net.getUnconnectedOutLayersNames())
        t2= time.time()
        out= outputs[0]
        n_detections= out.shape[1]
        height, width= img.shape[:2]
        x_scale= width/img_w
        y_scale= height/img_h
        conf_threshold= 0.7
        score_threshold= 0.5
        nms_threshold = 0.5

        class_ids=[]
        score=[]
        boxes=[]

        ngantuk_detected = False # Initialize the flag for ngantuk detection

        for i in range(n_detections):
            detect=out[0][i]
            confidence= detect[4]
            if confidence >= conf_threshold:
                class_score= detect[5:]
                class_id= np.argmax(class_score)
                if (class_score[class_id]> score_threshold):
                    score.append(confidence)
                    class_ids.append(class_id)
                    x, y, w, h = detect[0], detect[1], detect[2], detect[3]
                    left= int((x - w/2)* x_scale )
                    top= int((y - h/2)*y_scale)
                    width = int(w * x_scale)
                    height = int( h*y_scale)
                    box= np.array([left, top, width, height])
                    boxes.append(box)
                    
                    # Check if ngantuk is detected
                    if classes[class_id] == "kantuk":
                        ngantuk_detected = True

        indices = cv2.dnn.NMSBoxes(boxes, np.array(score), conf_threshold, nms_threshold)
        
        for i in indices:
            box = boxes[i]
            left = box[0]
            top = box[1]
            width = box[2]
            height = box[3]

            left, top, width, height = boxes[i]
            cv2.rectangle(img, (left, top), (left + width, top + height), (0, 0, 255), 2)
            label = "{}:{:.2f}".format(classes[class_ids[i]], score[i])
            # Modify the label to include the confidence score
            # Format the score to display only 2 decimal places
            text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)
            dim, baseline = text_size[0], text_size[1]
            cv2.rectangle(img, (left, top - 20), (left + dim[0], top + dim[1] + baseline - 20), (0,0,0), cv2.FILLED)
            cv2.putText(img, label, (left, top + dim[1] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1, cv2.LINE_AA)

            crop = img[top + 2:top + height - 2,left + 2:left + width - 2].copy()
            cv2.imshow('Detected Faces', crop)
            cv2.namedWindow("Detected Faces");
            cv2.moveWindow("Detected Faces", 670, 520);
        
        if len(score) > 0:
            score_label = "Score: {:.2f}".format(sum(score)/len(score))
            cv2.putText(img, score_label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

        
        height1 = int(img.shape[0] * (width_frame / img.shape[1]))
        dim1 = (width_frame, height1)
        img1 = cv2.resize(img, dim1, interpolation = cv2.INTER_AREA)
        cv2.imshow('Detection Info', img1)
        cv2.namedWindow("Detection Info");
        cv2.moveWindow("Detection Info", 630,0);
        # cv2.waitKey(0)

        if ngantuk_detected: # If ngantuk is detected, play a sound alert
            alert_sound.play()
            cv2.imwrite('result.jpg', img)

    width_frame = 600
    net = cv2.dnn.readNetFromONNX(os.path.join(model_directory, os.path.basename(model_path)))
    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
    net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
    classes = class_name()

    # Initialize pygame
    pygame.init()

    # Load alert sound
    alert_sound = pygame.mixer.Sound('alert.mp3')

    cap1 = cv2.VideoCapture(0, cv2.CAP_DSHOW)

    while True:
        ret1, frame1 = cap1.read()
        frame1 = cv2.flip(frame1,1)
        
        height1 = int(frame1.shape[0] * (width_frame / frame1.shape[1]))
        dim1 = (width_frame, height1)
        img1 = cv2.resize(frame1, dim1, interpolation = cv2.INTER_AREA)

        cv2.imshow('Realtime Feed', img1)
        cv2.namedWindow("Realtime Feed");
        cv2.moveWindow("Realtime Feed", 0, 0);

        detection1(frame1, net, classes)

        keyVal = cv2.waitKey(1) & 0xFF
        if keyVal == ord('x'):
            break
        elif keyVal == ord('q'):
            detection1(frame1, net, classes)

        # Exit the application if the "Esc" key is pressed
        if keyVal == 27:  # 27 is the ASCII code for "Esc" key
            break

    cap1.release()

    cv2.destroyAllWindows()
    root.deiconify()



detect = customtkinter.CTkButton(master=root,width=120,
                                 height=32,
                                 border_width=0,
                                 corner_radius=8,
                                 text="Detect",  
                                 command=run_detect)
detect.pack(padx=20, pady=10)
detect.place(x=925, y=500, anchor="n")

#Button Take Image
def dataprep():
    model_dataprep = Toplevel(root)  # Create a new Toplevel window
    model_dataprep.title("Data Preparation")
    model_dataprep.geometry('800x600')
    model_dataprep.state('zoomed')

    lokasi = filedialog.askdirectory(parent=model_dataprep,title="Select a directory")

    tabview = customtkinter.CTkTabview(model_dataprep)
    tabview.pack(fill="both", expand=True)

    tabview.add("DATASET")
    tabview.add("LABELING")
    tabview.add("HASIL")
    tabview.set("DATASET")

    def resize(directory):
    # Loop through all files in the directory
        for filename in os.listdir(directory):
            # Check if the file is a JPEG image
            if filename.lower().endswith((".jpg", ".jpeg")):
                # Open the image file
                filepath = os.path.join(directory, filename)
                image = Image.open(filepath)

                # Resize the image to fit within a 640x480 box
                image.thumbnail((640, 480))

                # Save the resized image
                new_filename = os.path.splitext(filename)[0] + "_resized.jpg"
                new_filepath = os.path.join(directory, new_filename)
                image.save(new_filepath)

                # Delete the original image file
                os.remove(filepath)

                # Print a message to indicate progress
                progress_label.configure(text=f"Resized {filename} to {new_filename} and deleted original")

    def select_and_resize():
        root = tk.Tk()
        root.withdraw()  # Hide the main window
        directory = lokasi
        if directory:
            resize(directory)
            progress_label.configure(text="Resizing process complete.")

    def labeling():
        subprocess.run(["python", "labelimg/labelimg.py"])

    def list_txt_files(directory):
        txt_files = []
    
    # Loop through all files in the directory
        for filename in os.listdir(directory):
            if filename.lower().endswith(".txt"):
                txt_files.append(filename)
        
        return txt_files

    def show_txt_content(selected_file):
        with open(selected_file, 'r') as file:
            content = file.read()
            content_window = tk.Toplevel()
            content_window.title("Isi File .txt")
            content_label = customtkinter.CTkLabel(master=content_window, text=content)
            content_label.pack(padx=20, pady=20)

    def hasil():
        directory = lokasi
        if directory:
            txt_files = list_txt_files(directory)
            if txt_files:
                result_text = "\n"
                for filename in txt_files:
                    result_text += filename + "\n"
            else:
                result_text = "Tidak ada file .txt dalam direktori tersebut."

            # Membuat canvas dengan scrolling

            canvas = tk.Canvas(tabview.tab("HASIL"))
            canvas.pack(fill=tk.BOTH, expand=True)

            scrollbar = tk.Scrollbar(canvas, command=canvas.yview)
            scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
            canvas.config(yscrollcommand=scrollbar.set)

            frame = tk.Frame(canvas)
            canvas.create_window((0, 0), window=frame, anchor=tk.N)

            for filename in txt_files:
                file_button = customtkinter.CTkButton(frame, text=f"{filename}", command=lambda name=filename: show_txt_content(os.path.join(directory, name)))
                file_button.pack(padx=5, pady=5)
                
            frame.update_idletasks()
            canvas.config(scrollregion=canvas.bbox("all"))

            # Configure canvas scrolling behavior
            canvas.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))

    def close_prep_window():
        model_dataprep.destroy()
        
    # eval_label = tk.Label(eval_window, text="Evaluation Window using Custom Frame")
    # eval_label.pack()
    # eval_button = Button(eval_window, text="Select Weights File", font=50, command=select_weights_file)
    # eval_button.pack()
    # eval_button.place(relx=0.5, y=100, anchor="n")

    # Button to close the eval_window
    close_button = customtkinter.CTkButton(master=model_dataprep, text="Close", command=close_prep_window)
    close_button.pack()
    eval_button.place(relx=0.5, y=500, anchor="n")

    progress_label = customtkinter.CTkLabel(tabview.tab("DATASET"), text="Masukkan Data", wraplength=300)
    progress_label.pack(padx=20, pady=10)

    button_1 = customtkinter.CTkButton(tabview.tab("DATASET"), text="Resize Dataset",command=select_and_resize)
    button_1.pack(padx=20, pady=20)

    button_2 = customtkinter.CTkButton(tabview.tab("LABELING"), text="Resize Dataset",command=labeling)
    button_2.pack(padx=20, pady=20)

    button_2 = customtkinter.CTkButton(tabview.tab("HASIL"), text="TAMPILKAN HASIL",command=hasil)
    button_2.pack(padx=20, pady=20)




takeimg = customtkinter.CTkButton(master=root,width=120,
                                 height=32,
                                 border_width=0,
                                 corner_radius=8, text="Data Preparation", 
                                 command=dataprep)
takeimg.pack(padx=20, pady=10)
takeimg.place(x=615, y=500, anchor="n")

def run_export():
    root = Tk()
    root.withdraw()  # Hide the root window temporarily
    
    initial_dir = os.path.join("yolov5", "runs", "train")
    weights_path = filedialog.askopenfilename(title="Select Weights File", initialdir=initial_dir, filetypes=[("Weights files", "*.pt")])

    if not weights_path:
        print("No weights file selected. Exiting...")
        return
    
    # Extract the path starting from "yolov5"
    extracted_path = weights_path.split("train", 1)[-1].lstrip("\\/")


    command = f"python export.py --weights runs/train/{extracted_path} --include torchscript onnx"
    subprocess.run(f'start /wait cmd /k "{command}"', shell=True)


# export_button = Button(root, text="Export Weights", font=50, command=run_export)
# export_button.pack()
# export_button.place(relx=0.5, y=560, anchor="n")

def run_eval(): 
    eval_window = Toplevel(root)  # Create a new Toplevel window
    eval_window.title("Evaluation")
    eval_window.geometry('800x600')
    eval_window.state('zoomed')

    tabview = customtkinter.CTkTabview(eval_window)
    tabview.pack(fill="both", expand=True)
    
    def select_weights_file():
        initialeval_dir = os.path.join("yolov5", "runs", "train")
        weightseval_path = filedialog.askopenfilename(parent=eval_window, title="Select Weights File", initialdir=initialeval_dir, filetypes=[("Weights files", "*.pt")])

        
        if not weightseval_path:
            print("No weights file selected. Exiting...")
            return
        
        # Extract the path starting from "yolov5"
        extractedeval_path = weightseval_path.split("train", 1)[-1].lstrip("\\/")


        commandeval = f"python val.py --data custom_dataset.yaml --weights runs/train/{extractedeval_path}"
        subprocess.run(f'start /wait cmd /k "{commandeval}"', shell=True)

        notification_title = "Evaluation Completed"
        notification_text = "The evaluation process has completed."
        notification.notify(title=notification_title, message=notification_text)
    
    def close_eval_window():
        eval_window.destroy()
        
    # eval_label = tk.Label(eval_window, text="Evaluation Window using Custom Frame")
    # eval_label.pack()
    # eval_button = Button(eval_window, text="Select Weights File", font=50, command=select_weights_file)
    # eval_button.pack()
    # eval_button.place(relx=0.5, y=100, anchor="n")

    # Button to close the eval_window
    close_button = customtkinter.CTkButton(master=eval_window, text="Close", command=close_eval_window)
    close_button.pack()
    eval_button.place(relx=0.5, y=500, anchor="n")  

    tabview.add("INPUT DATASET")
    tabview.add("EVALUATION")
    tabview.add("HASIL")
    tabview.set("EVALUATION")

    button_1 = customtkinter.CTkButton(tabview.tab("EVALUATION"), text="Pilih Model", command=select_weights_file)
    button_1.pack(padx=20, pady=20)

    model_label = tk.Label(tabview.tab("EVALUATION"), text="Evaluation Window using Custom Frame")
    model_label.pack()
    
#Button for opening the evaluation window
eval_button = customtkinter.CTkButton(master=root,width=120,
                                 height=32,
                                 border_width=0,
                                 corner_radius=8, text="Evaluation", 
                                 command=run_eval)
eval_button.pack(padx=20, pady=10)
eval_button.place(relx=0.5, y=500, anchor="n")

   


def main():
    root.mainloop()

if __name__ == "__main__":
    main()