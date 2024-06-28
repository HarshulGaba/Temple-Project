from yolov5.utils.torch_utils import *
from yolov5.utils.general import *
from yolov5.models.experimental import *
import cv2
import torch


# Load model
model = attempt_load('yolov5s.pt', device=torch.device('cpu'))

# Set model to evaluation mode
model.eval()

# Open video stream
cap = cv2.VideoCapture(1)

while True:
    # Read frame from video stream
    ret, frame = cap.read()

    # Resize frame to model input size
    img = cv2.resize(frame, (640, 640))

    # Convert frame from BGR to RGB
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Inference
    results = model(img)

    # Draw bounding boxes
    for det in results.pred[0]:
        # Extract coordinates and class ID
        x1, y1, x2, y2, conf, cls = det.cpu().numpy()

        # Draw bounding box
        label = f'{int(cls)} {conf:.2f}'
        color = (0, 255, 0)  # Green
        thickness = 2
        cv2.rectangle(frame, (int(x1), int(y1)),
                      (int(x2), int(y2)), color, thickness)
        cv2.putText(frame, label, (int(x1), int(y1 - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, thickness)

    # Display output frame
    cv2.imshow('frame', frame)

    # Exit on ESC key press
    if cv2.waitKey(1) == 27:
        break

# Release video stream and close all windows
cap.release()
cv2.destroyAllWindows()
