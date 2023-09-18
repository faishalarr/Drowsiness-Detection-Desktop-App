import subprocess

# Command to run
command = "python export.py --weights runs/train/exp/weights/best.pt --include torchscript onnx"

# Open cmd and run the command
# Open cmd and run the command
subprocess.run(f'start /wait cmd /k "{command}"', shell=True)