import cv2
from models.experimental import attempt_load
from utils.general import non_max_suppression, scale_coords

weights = 'best.pt'
device = 'cpu'
model = attempt_load(weights, map_location=device)

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Preprocess the input image (resize, normalize, etc.)
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (640, 640))
    img = img / 255.0
    img = img.transpose((2, 0, 1))
    img = img.astype('float32')
    img = img.reshape(1, 3, 640, 640)

    # Perform object detection
    results = model(img, size=640)
    results = non_max_suppression(results, conf_thres=0.5, iou_thres=0.5)

    # Display the results
    for result in results:
        for det in result:
            box = det[:4]
            score = det[4]
            label = int(det[5])
            color = (0, 255, 0)
            bbox = cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), color, 2)
            cv2.putText(bbox, f'{score:.2f}', (box[0], box[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    cv2.imshow('Object Detection', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
