import cv2
import socket
import pickle
import struct

import numpy as np
from PIL import Image

from client_util import videoDetector

client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
host_ip = ''  # your server ip
port = 9999  # your socket communication port
client_socket.connect((host_ip, port))

cap = cv2.VideoCapture(cv2.CAP_DSHOW + 0)

# set webcam resolution as 512x512
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 512)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 512)
data = b""
payload_size = struct.calcsize("Q")
while cap.isOpened():
    ret, frame = cap.read()
    if ret:
        prompt = videoDetector(frame)
        prompt += "best quality, marvel, hero, happy"
        print(prompt)
        frame_cvt = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_pil = Image.fromarray(frame_cvt)
        combined_data = {'image': frame_pil, 'text': prompt}
        a = pickle.dumps(combined_data)
        message = struct.pack("Q", len(a)) + a
        client_socket.sendall(message)

        cv2.imshow('Sending Video', frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break

        fps = cap.get(cv2.CAP_PROP_FPS)
        print(f"Current FPS: {fps}")

        while len(data) < payload_size:
            packet = client_socket.recv(4 * 1024)  # buffer size 4K
            if not packet: break
            data += packet
        packed_msg_size = data[:payload_size]
        data = data[payload_size:]
        msg_size = struct.unpack("Q", packed_msg_size)[0]

        # get data frame
        while len(data) < msg_size:
            data += client_socket.recv(4 * 1024)
        frame_data = data[:msg_size]
        data = data[msg_size:]

        # get transformed image
        frame = pickle.loads(frame_data)
        frame = cv2.cvtColor(np.array(frame), cv2.COLOR_RGB2BGR)
        cv2.imshow('Received Transformed Video', frame)
    else:
        break

client_socket.close()
cap.release()
cv2.destroyAllWindows()
