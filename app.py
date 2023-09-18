import cv2
import torch
from yolov5.models.experimental import attempt_load
from yolov5.utils.general import non_max_suppression, scale_coords
from yolov5.utils.torch_utils import select_device

# Load the YOLOv5 model
device = select_device('')
model = attempt_load('yolov5\runs\train\exp\weights\best.pt', map_location=device)

# Set the model to inference mode
model.eval()

# Define the confidence threshold and the IoU threshold for NMS
conf_threshold = 0.5
iou_threshold = 0.5

# Define the colors for the bounding boxes
colors = [(0, 255, 0), (0, 0, 255), (255, 0, 0)]

# Define the function to perform object detection on each frame
def detect(frame):
    # Resize the frame to the input size of the model
    input_size = model.img_size
    frame_resized = cv2.resize(frame, input_size)

    # Convert the frame from BGR to RGB and normalize it
    frame_norm = frame_resized[:, :, ::-1].transpose(2, 0, 1) / 255.0

    # Convert the frame to a Torch tensor and add a batch dimension
    frame_tensor = torch.from_numpy(frame_norm).unsqueeze(0).float()

    # Move the tensor to the device and run it through the model
    device = select_device('')
    model.to(device)
    frame_tensor = frame_tensor.to(device)
    with torch.no_grad():
        outputs = model(frame_tensor)

    # Post-process the outputs and extract the bounding boxes, labels, and scores
    outputs = non_max_suppression(outputs, conf_threshold, iou_threshold, classes=None, agnostic=False)
    bboxes = []
    labels = []
    scores = []
    for output in outputs:
        if output is not None and len(output) > 0:
            output = output[0]
            bboxes.append(scale_coords(input_size, output[:, :4], frame.shape[:2]).round())
            labels.append(output[:, 5].long())
            scores.append(output[:, 4])

    # Draw the bounding boxes on the frame
    for i, bbox in enumerate(bboxes):
        color = colors[i % len(colors)]
        for x1, y1, x2, y2 in bbox:
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            label = f'class {labels[i][0]}'
            score = f'{scores[i][0]:.2f}'
            cv2.putText(frame, f'{label} {score}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    return frame

# Open the video capture device
cap = cv2.VideoCapture(0)

# Loop over the frames and perform object detection on each frame
while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame = detect(frame)
    cv2.imshow('Object Detection', frame)
    if cv2.waitKey(1) == ord('q'):
        break

# Release the video capture device and close the windows
cap.release()
cv2.destroyAllWindows()
