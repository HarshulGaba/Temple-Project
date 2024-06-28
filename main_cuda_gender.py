import torch
import cv2
import time

# Load YOLOv5 object detection model
model = torch.hub.load('ultralytics/yolov5', 'yolov5x', pretrained=True)
gender_model = cv2.dnn.readNetFromCaffe(
    "models/gender_deploy.prototxt", "models/gender_net.caffemodel")


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

    # Loop through detected objects and classify gender
    for result in results.xyxy[0]:
        if result[5] == 0:  # person class
            x1, y1, x2, y2 = result[:4]
            x1 = int(x1)
            y1 = int(y1)
            x2 = int(x2)
            y2 = int(y2)
            face = img[y1:y2, x1:x2]  # crop person image
            face_blob = cv2.dnn.blobFromImage(cv2.resize(face, (227, 227)), 1.0, (
                227, 227), (78.4263377603, 87.7689143744, 114.895847746), swapRB=True)
            gender_model.setInput(face_blob)  # predict gender
            gender_pred = gender_model.forward()
            gender = 'Male' if gender_pred.argmax() == 0 else 'Female'
            # Draw box and label on image
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(img, gender, (x1, y1-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    # Show image
    cv2.imshow('Image', img)

    # Exit on ESC key press
    if cv2.waitKey(1) == 27:
        break

# Release video stream and close all windows
cap.release()
cv2.destroyAllWindows()
