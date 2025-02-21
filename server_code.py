from flask import Flask, request, Response
import cv2
import base64
import numpy as np
import datetime
from ultralytics import YOLO
import sys
import torch


app = Flask(__name__)

received_data = {'img_base64': ''}

#watermark_path = 'image.jpeg'
#watermark = cv2.imread(watermark_path, cv2.IMREAD_UNCHANGED)
model = torch.hub.load('ultralytics/yolov5', 'custom',path = '/home/tihan_husky/Downloads/Client/human/best.pt')

@app.route('/receive_data_image', methods=['POST'])
def receive_data_image():
    global received_data

    if request.method == 'POST':
        data = request.json

        received_data['img_base64'] = data['img']

    return 'Data received successfully'

def generate():
    global received_data

    while True:
        img_data = base64.b64decode(received_data['img_base64'])
        nparr = np.frombuffer(img_data, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        results = model(img)
        drone_detected = False
        for detection in results.xyxy[0]:
            xmin, ymin, xmax, ymax,conf,cls = detection.tolist()
            class_name=results.names[int(cls)]
            cv2.rectangle(img, (int(xmin), int(ymin)), (int(xmax), int(ymax)), (0,0,255),4)
            cv2.putText(img,f":{class_name}- :{conf:.2f}", (int(xmin), int(ymin)-5), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0,0,255), 4)
            if class_name == 'drone':
               drone_detected = True
               break
        if drone_detected == True:
            cv2.putText(img, 'DRONE DETECTED', (10,230), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0,0,255), 2)
 #       watermark_width = 100
 #       watermark_resized = cv2.resize(watermark, (watermark_width, int(watermark_width * watermark.shape[0] / watermark.shape[1])))
        frame_height, frame_width, _ = img.shape

 #       watermark_position = (frame_width - watermark_resized.shape[1], frame_height - watermark_resized.shape[0])

#        img[watermark_position[1]:watermark_position[1] + watermark_resized.shape[0],
#            watermark_position[0]:watermark_position[0] + watermark_resized.shape[1]] = overlay_transparent(
#            img[watermark_position[1]:watermark_position[1] + watermark_resized.shape[0],
#            watermark_position[0]:watermark_position[0] + watermark_resized.shape[1]],
#            watermark_resized)

        _, img_encoded = cv2.imencode('.jpg', img)
        img_bytes = img_encoded.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + img_bytes + b'\r\n\r\n')

def overlay_transparent(background, overlay, position=(0, 0)):
    h, w, _ = overlay.shape

    overlay_image = overlay[..., :3]
    mask = overlay[..., 3:] / 255.0

    background[position[1]:position[1] + h, position[0]:position[0] + w] = \
        (1.0 - mask) * background[position[1]:position[1] + h, position[0]:position[0] + w] + mask * overlay_image

    return background

@app.route('/termal')
def video_feed():
    return Response(generate(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080, debug=True)
