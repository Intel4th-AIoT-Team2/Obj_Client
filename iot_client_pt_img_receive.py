import cv2
from sympy import im
from ultralytics import YOLO
import socket
import sys
from queue import Queue
import numpy as np
from threading import Thread

# YOLO model initialization
model = YOLO('fire.pt')

# Confidence threshold (80%)
CONF_THRESHOLD = 0.80

# Queue for image frames
img_queue = Queue()

# Socket setup
def setup_socket(ip, port, name):
    # connect to server
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.connect((ip, port))
        sock.sendall(f"[{name}:PASSWD]\n".encode())
        return sock
    except socket.error as e:
        print(f"Socket error: {e}")
        exit(1)


# image receive thread
def receive_img(sock):
    # 이미지 데이터는 다음과 같은 방식으로 전송됨:
    # client_socket.sendall((str(len(byteData))).encode().ljust(16) + byteData)
    while True:
        try:
            buf = b''

            # Read image size
            img_size = sock.recv(16).decode().strip()
            if not img_size:
                break
            img_size = int(img_size)

            # Read image data
            while len(buf) < img_size:
                data = sock.recv(img_size - len(buf))
                if not data:
                    break
                buf += data
            
            # Convert byte data to numpy array
            byte_data = np.frombuffer(buf, dtype=np.uint8)

            # Decode image
            img = cv2.imdecode(byte_data, cv2.IMREAD_COLOR)
            img_queue.put(img)
        except Exception as e:
            print(f"Error receiving image: {e}")
            break
    sock.close()


# Object detection and sending coordinates
def detect_and_send(sock, name):
    while True:
        # Get image from queue
        if img_queue.empty():
            continue
        frame = img_queue.get()

        results = model(frame)
        annotated_frame = frame

        for result in results:
            try:
                annotated_frame = result.plot()
                for box in result.boxes:
				    # Confidence score for the current box
                    confidence = box.conf[0].item()

                    # Process only if confidence is above threshold
                    if confidence >= CONF_THRESHOLD:
                        x1, y1, x2, y2 = box.xyxy[0]
                        center_x = int((x1 + x2) / 2)
                        center_y = int((y1 + y2) / 2)

                        class_id = int(box.cls[0])
                        class_name = model.names[class_id]

                        print(f"Object detected: {class_name} at ({center_x}, {center_y})")

                        msg = f"[ALLMSG]{class_name}@{center_x}@{center_y}\n"
                        try:
                            sock.sendall(msg.encode())
                        except socket.error as e:
                            print(f"Send error: {e}")
                            # cap.release()
                            sock.close()
                            exit(1)

                        # Draw circle on detected object
                        cv2.circle(annotated_frame, (center_x, center_y), 5, (0, 255, 0), -1)

            except Exception as e:
                print(f"Error plotting results: {e}")

        cv2.imshow("YOLOv8 Real-Time Detection", annotated_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # cap.release()
    cv2.destroyAllWindows()
    sock.close()

def main():
    if len(sys.argv) != 4:
        print(f"Usage: {sys.argv[0]} <IP> <port> <name>")
        exit(1)

    ip = sys.argv[1]
    port = int(sys.argv[2])
    name = sys.argv[3]

    # Set up socket and establish connection
    sock = setup_socket(ip, port, name)

    # Set up socket for image receive
    obj_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    obj_sock.bind(("0.0.0.0", 5001))
    print("waiting for connection")
    obj_sock.listen(1)
    pi_sock, _ = obj_sock.accept()
    print("connected")

    # send connect message to pi_sock
    pi_sock.sendall("서버와 연결되었습니다.".encode("utf-8"))

    # Start image receive thread
    img_thread = Thread(target=receive_img, args=(pi_sock,))
    img_thread.daemon = True
    img_thread.start()

    # Start object detection after successful connection
    detect_and_send(sock, name)

if __name__ == "__main__":
    main()
