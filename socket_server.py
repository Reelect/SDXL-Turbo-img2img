import socket
import pickle
import struct
from diffusers import AutoPipelineForImage2Image
import torch

pipe = AutoPipelineForImage2Image.from_pretrained("stabilityai/sdxl-turbo", torch_dtype=torch.float16, variant="fp16")
pipe.to("cuda")


def img2img(img):
    img.resize((512, 512))
    prompt = "cat wizard, gandalf, lord of the rings, detailed, fantasy, cute, adorable, Pixar, Disney, 8k"
    image = pipe(prompt, image=img, num_inference_steps=2, strength=0.5, guidance_scale=0.0).images[0]
    return image


server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
host_ip = '0.0.0.0'
port = 9999
server_socket.bind((host_ip, port))
server_socket.listen(5)
print('Listening at:', (host_ip, port))

while True:
    client_socket, addr = server_socket.accept()
    data = b""
    payload_size = struct.calcsize("Q")
    while True:
        while len(data) < payload_size:
            packet = client_socket.recv(4 * 1024)
            if not packet: break
            data += packet
        packed_msg_size = data[:payload_size]
        data = data[payload_size:]
        msg_size = struct.unpack("Q", packed_msg_size)[0]

        while len(data) < msg_size:
            data += client_socket.recv(4 * 1024)
        frame_data = data[:msg_size]
        data = data[msg_size:]
        frame = pickle.loads(frame_data)

        # img2img 처리
        processed_frame = img2img(frame)

        # 변환된 프레임을 클라이언트에 전송
        frame_data = pickle.dumps(processed_frame)
        message = struct.pack("Q", len(frame_data)) + frame_data
        client_socket.sendall(message)
    client_socket.close()