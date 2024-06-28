import torch
import cv2
import time
import numpy as np
# Load YOLOv5 model
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

# Start video capture
cap = cv2.VideoCapture(0)

while True:
    # Read frame from video stream
    ret, frame = cap.read()

    # Resize frame to model input size
    img = cv2.resize(frame, (640, 640))

    # Convert frame from BGR to RGB
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Run object detection
    results = model(img)

    # Draw bounding boxes on image
    img = results.render()

    # Show image
    cv2.imshow("VIDEO", np.asarray(img)[0])

    print(results.xyxy[0])
    # Exit on ESC key press
    if cv2.waitKey(10) == 'q':
        break

# Release video stream and close all windows
cap.release()
cv2.destroyAllWindows()


# # Results
# results.print()
# results.save()  # or .show()

# results.xyxy[0]  # img1 predictions (tensor)
# results.pandas().xyxy[0]  # img1 predictions (pandas)
# #      xmin    ymin    xmax   ymax  confidence  class    name
# # 0  749.50   43.50  1148.0  704.5    0.874023      0  person
# # 1  433.50  433.50   517.5  714.5    0.687988     27     tie
# # 2  114.75  195.75  1095.0  708.0    0.624512      0  person
# # 3  986.00  304.00  1028.0  420.0    0.286865     27     tie
