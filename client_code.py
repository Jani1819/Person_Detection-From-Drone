import cv2
import requests
import numpy as np
import threading
import json
import base64
import asyncio

def capture_and_transmit():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open camera.")
        return

    server_address = 'http://192.168.20.232:8080/receive_data_image'  # Replace <Machine_2_IP> with the IP address of Machine 2

    while True:
        ret, frame = cap.read()
        resized_frame = cv2.resize(frame, (1000, 800 ))
        _, img_encoded = cv2.imencode('.jpg', resized_frame)
        img_bytes = img_encoded.tobytes()
        img_base64 = base64.b64encode(img_bytes).decode('utf-8')

        data = {
                'img': img_base64
        }
        try:
            response = requests.post(server_address, json=data)
            print(f"Response from server: {response.text}")
        except requests.exceptions.RequestException as e:
            print(f"Error sending data: {e}")

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()

capture_thread = threading.Thread(target=capture_and_transmit)
capture_thread.start()

capture_thread.join()

cv2.destroyAllWindows()