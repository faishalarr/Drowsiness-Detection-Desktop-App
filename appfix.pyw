from tkinter import *
from PIL import ImageTk, Image
import subprocess
from datetime import date
from datetime import datetime


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
    subprocess.run(["python", "D:/ical/Magang Astra/Object-Detection-using-Yolov5-main/Object Detection using Yolov5/yolov5/runs/train/exp2/weights/choose_model.pyw"])

cuy = Label(image=det1)
cuy.pack()
cuy.place(relx=0.5,x= -70, y=370, anchor="n")

detect = Button(root, text="Detect", font=50, command=run_detect)
detect.pack()
detect.place(relx=0.5, y=380, anchor="n")

#Button Take Image
def run_takeimg():
    subprocess.run(["python", "latihmodel.pyw"])

cuy1 = Label(image=cam1)
cuy1.pack()
cuy1.place(relx=0.5,x= -100, y=430, anchor="n")

takeimg = Button(root, text="Take Image", font=50, command=run_takeimg)
takeimg.pack()
takeimg.place(relx=0.5, y=440, anchor="n")


def main():
    root.mainloop()

if __name__ == "__main__":
    main()