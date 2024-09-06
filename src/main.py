from io import BytesIO
from socket import socket, AF_UNIX, SOCK_STREAM
from sys import byteorder

from pipelines.models import TextToImageRequest

from pipeline import load_pipeline, infer

SOCKET = "/api/inferences.sock"


def main():
    pipeline = load_pipeline()

    with socket(AF_UNIX, SOCK_STREAM) as inference_socket:
        inference_socket.bind(SOCKET)

        inference_socket.listen(1)
        connection, _ = inference_socket.accept()

        with connection:
            connection.send(b'\xFF') # Ready marker

            while True:
                size = int.from_bytes(connection.recv(2), byteorder)

                request = TextToImageRequest.model_validate_json(connection.recv(size).decode("utf-8"))

                image = infer(request, pipeline)

                data = BytesIO()
                image.save(data, format=image.format)

                packet = data.getvalue()

                connection.send(len(packet).to_bytes(4, byteorder))
                connection.send(packet)


if __name__ == '__main__':
    main()
