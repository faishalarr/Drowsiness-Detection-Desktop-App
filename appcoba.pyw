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

try:
    subprocess.check_call(["pip", "install", "-r", "requirements.txt"])
except subprocess.CalledProcessError:
    print("An error occurred while installing required packages.")


root = Tk()
root.title("Spec Detection")
root.iconbitmap("D:/ical/Magang Astra/Object-Detection-using-Yolov5-main/Object Detection using Yolov5/yolov5/runs/train/exp2/weights/Daihatsu.ico")
root.geometry('1920x1080')
root.state('zoomed')

#Hari
today = date.today()	
d2 = today.strftime("%B %d, %Y")


#Waktu
time = datetime.now()
rn = time.strftime("%I:%M:%S %p")


#gambar daihatsu
img0 = Image.open('D:/ical/Magang Astra/SPEK/Daihatsu.jpg')
logo0 = img0.resize((200,160),Image.ANTIALIAS)
logo1 = ImageTk.PhotoImage(logo0)

#gambar icon 1
img1 = Image.open('D:/ical/Magang Astra/SPEK/camera.png')
cam = img1.resize((50,50),Image.ANTIALIAS)
cam1 = ImageTk.PhotoImage(cam)

#gambar icon 2
img2 = Image.open('D:/ical/Magang Astra/SPEK/loupe.png')
det = img2.resize((50,50),Image.ANTIALIAS)
det1 = ImageTk.PhotoImage(det)

title = Label(text="SPEC DETECTION", font=('Arial',30))
title.pack()
title.place(relx=0.5,anchor="n",y=20)

up = Label(image=logo1)
up.pack(side='left', anchor='nw')


tgl = Label(text=d2,font=('Arial',16))
tgl.pack(side='top', anchor='ne')

jam = Label(text=rn,font=('Arial',16))
jam.pack(side='top', anchor='ne',pady=10)


def update_time():
    now = datetime.now()
    tgl.config(text=now.strftime("%d %B, %Y"))
    jam.config(text=now.strftime("%H:%M:%S"))
    root.after(1000, update_time)  # call update_time() again in 1 second

update_time() 

# w = (root.winfo_screenwidth() - root.winfo_reqwidth()) / 2
# detect_x = w - 50
# takeimg_x = w + 50

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

cuy = Label(image=det1)
cuy.pack()
cuy.place(relx=0.5,x= -70, y=370, anchor="n")

detect = Button(root, text="Detect", font=50, command=run_detect)
detect.pack()
detect.place(relx=0.5, y=380, anchor="n")

#Button Take Image
def run_takeimg():
    command = "python train.py --img 640 --batch 16 --epochs 100 --data custom_dataset.yaml --weights yolov5s.pt --cache"
    subprocess.run(f'start /wait cmd /k "{command}"', shell=True)

cuy1 = Label(image=cam1)
cuy1.pack()
cuy1.place(relx=0.5,x= -100, y=430, anchor="n")

takeimg = Button(root, text="Take Image", font=50, command=run_takeimg)
takeimg.pack()
takeimg.place(relx=0.5, y=440, anchor="n")

def run_tambahdataset():
    subprocess.run(["python", "tambahdata.py"])
# def run_tambahdataset(destination_folder):
#     file_paths = filedialog.askopenfilenames(filetypes=[("Image files", "*.jpg *.jpeg *.png")])

#     for file_path in file_paths:
#         file_name = os.path.basename(file_path)
#         destination_path = os.path.join(destination_folder, file_name)
#         shutil.copy(file_path, destination_path)
#         print(f"Added {file_name} to {destination_folder}")

# def run_tambahdata_both():
#     default_destination_train = 'data/train/images'
#     default_destination_val = 'data/val/images'

#     run_tambahdataset(default_destination_train)
#     run_tambahdataset(default_destination_val)

# tambahdata_both = Button(root, text="Add Data to Train and Val", font=50, command=run_tambahdata_both)
# tambahdata_both.pack()
# tambahdata_both.place(relx=0.5, y=20, anchor="n")

cuy2 = Label(image=cam1)
cuy2.pack()
cuy2.place(relx=0.5,x= -100, y=490, anchor="n")

takeimg = Button(root, text="Take Image", font=50, command=run_tambahdataset)
takeimg.pack()
takeimg.place(relx=0.5, y=500, anchor="n")


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


export_button = Button(root, text="Export Weights", font=50, command=run_export)
export_button.pack()
export_button.place(relx=0.5, y=560, anchor="n")

def run_eval():
    eval_window = Toplevel(root)  # Create a new Toplevel window
    eval_window.title("Evaluation")
    eval_window.geometry('800x600')
    eval_window.state('zoomed')
    
    def select_weights_file():
        initialeval_dir = os.path.join("yolov5", "runs", "train")
        weightseval_path = filedialog.askopenfilename(title="Select Weights File", initialdir=initialeval_dir, filetypes=[("Weights files", "*.pt")])

        
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
        
        
    
    eval_button = Button(eval_window, text="Select Weights File", font=50, command=select_weights_file)
    eval_button.pack()
    eval_button.place(relx=0.5, y=100, anchor="n")



# Button for opening the evaluation window
eval_button = Button(root, text="Evaluation", font=50, command=run_eval)
eval_button.pack()
eval_button.place(relx=0.5, y=620, anchor="n")


def main():
    root.mainloop()

if __name__ == "__main__":
    main()