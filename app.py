from flask import Flask, render_template, request, redirect
import cv2
import numpy as np
from ultralytics import YOLO
import base64

app = Flask(__name__)


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/upload', methods=['POST'])
def upload():
    # Check if the POST request has the file part
    if 'file' not in request.files:
        return redirect(request.url)

    file = request.files['file']

    if file.filename == '':
        return redirect(request.url)

    # Read the image file
    img_array = np.frombuffer(file.read(), dtype=np.uint8)
    img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

    # Perform person detection using YOLO
    model = YOLO('yolov5s.pt')
    result = model(source=img, conf=0.8)

    # Iterate over each detection result
    for detection in result:
        # Extract bounding box coordinates from the 'boxes' attribute
        boxes = detection.boxes.xyxy
        for box in boxes:
            # Extract coordinates
            x_min, y_min, x_max, y_max = map(int, box)
            # Draw bounding box
            cv2.rectangle(img, (x_min, y_min), (x_max, y_max), (0, 255, 0), 4)

    # Encode image as JPG for displaying in HTML
    _, img_encoded = cv2.imencode('.jpg', img)
    img_str = base64.b64encode(img_encoded).decode('utf-8')

    return render_template('result.html', img_data=img_str)


if __name__ == '__main__':
    app.run(debug=True)
