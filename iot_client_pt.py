import cv2
from ultralytics import YOLO
import socket
import sys

# YOLO model initialization
model = YOLO('fire.pt')

# Confidence threshold (80%)
CONF_THRESHOLD = 0.80

# Socket setup
def setup_socket(ip, port, name):
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.connect((ip, port))
        sock.sendall(f"[{name}:PASSWD]\n".encode())
        return sock
    except socket.error as e:
        print(f"Socket error: {e}")
        exit(1)

# Object detection and sending coordinates
def detect_and_send(sock, name):
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open video capture.")
        exit(1)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

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
                            cap.release()
                            sock.close()
                            exit(1)

                        # Draw circle on detected object
                        cv2.circle(annotated_frame, (center_x, center_y), 5, (0, 255, 0), -1)

            except Exception as e:
                print(f"Error plotting results: {e}")

        cv2.imshow("YOLOv8 Real-Time Detection", annotated_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
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

    # Start object detection after successful connection
    detect_and_send(sock, name)

if __name__ == "__main__":
    main()
