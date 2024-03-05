import socket
import pickle
import struct
from diffusers import AutoPipelineForImage2Image
import torch
import os

# seed fix
os.environ["CUBLAS_WORKSPACE_CONFIG"]=":16:8"
torch.backends.cudnn.benchmark = False
torch.use_deterministic_algorithms(True)
g = torch.Generator(device="cuda")

# if you own your model repository, write the directory path instead of the below model string
pipe = AutoPipelineForImage2Image.from_pretrained("stabilityai/sdxl-turbo", torch_dtype=torch.float16, variant="fp16")
pipe.to("cuda")


def img2img(img, custom_prompt):
    img.resize((512, 512))
    # set your prompt for generation
    # prompt = "cat wizard, gandalf, lord of the rings, detailed, fantasy, cute, adorable, Pixar, Disney, 8k"
    # customize your generation option before execute this server file
    g.manual_seed(42) # you can change your desire seed
    image = pipe(custom_prompt, image=img, num_inference_steps=2,  generator=g, strength=0.5, guidance_scale=0.0).images[0]
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
        received_data = pickle.loads(frame_data)
        frame = received_data['image']  # PIL Image
        prompt = received_data['text']  # Additional text information

        # img2img generating
        processed_frame = img2img(frame, prompt)

        # respond to client with transformed image
        frame_data = pickle.dumps(processed_frame)
        message = struct.pack("Q", len(frame_data)) + frame_data
        client_socket.sendall(message)
    client_socket.close()
