import subprocess

# Command to run
command = "python train.py --img 640 --batch 16 --epochs 100 --data custom_dataset.yaml --weights yolov5s.pt --cache"

# Open cmd and run the command
# Open cmd and run the command
subprocess.run(f'start /wait cmd /k "{command}"', shell=True)