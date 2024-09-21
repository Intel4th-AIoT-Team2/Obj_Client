import sys
import socket
import struct
from threading import Thread

import cv2
import numpy as np
from sympy import im
from ultralytics import YOLO



SERVER_IP = "10.10.141.153"     # Qt server ip
SERVER_PORT = 5020              # Qt server port
CAM1_PORT = 5021                # Cam1 image port
CAM2_PORT = 5022                # Cam2 image port

cam1_image = None
cam1_ready = False
cam1_count = 0

CONF_THRESHOLD = 0.80           # Confidence threshold

model = YOLO('fire.pt')      # YOLO model

# 소켓 설정
def setup_socket(ip, port, name):
    # 서버에 연결
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.connect((ip, port))
        sock.sendall(f"[{name}:PASSWD]\n".encode())
        return sock
    except socket.error as e:
        print(f"소켓 에러: {e}")
        exit(1)


# 이미지 수신 스레드
def receive_img(sock):
    global cam1_image, cam1_ready, cam1_count

    # 이미지 데이터는 다음과 같은 방식으로 전송됨:
    # client_socket.sendall((str(len(byteData))).encode().ljust(16) + byteData)
    
    try:
        while True:
            try:
                buf = b''

                # 이미지 크기 읽기
                img_size = sock.recv(16).decode().strip()
                if not img_size:
                    break
                img_size = int(img_size)

                # 이미지 데이터 읽기
                while len(buf) < img_size:
                    data = sock.recv(img_size - len(buf))
                    if not data:
                        break
                    buf += data
                
                # 바이트 데이터를 numpy 배열로 변환
                byte_data = np.frombuffer(buf, dtype=np.uint8)

                # 이미지 디코딩
                cam1_image = cv2.imdecode(byte_data, cv2.IMREAD_COLOR)
                cam1_count += 1

            except BrokenPipeError:
                print(f"[Error] Client disconnected")
                
            except Exception as e:
                print(f"이미지 수신 오류: {e}")
    finally:
        if sock is not None:
            sock.close()
        cam1_ready = False


# 객체 감지 및 좌표 전송
def detect_and_send(sock):
    global cam1_image, cam1_ready, cam1_count

    last_count = -1
    
    while cam1_ready is True:
        # 큐에서 이미지 가져오기
        if cam1_image is None or cam1_count == last_count:
            continue

        # YOLO 모델을 사용해 객체 감지
        results = model(cam1_image, verbose=False)
        annotated_frame = cam1_image
        last_count = cam1_count

        for result in results:
            try:
                # 결과 이미지 그리기
                annotated_frame = result.plot()
                for box in result.boxes:
				    # 현재 박스의 신뢰도 점수
                    confidence = box.conf[0].item()

                    # 신뢰도가 임계값 이상일 때만 처리
                    if confidence >= CONF_THRESHOLD:
                        x1, y1, x2, y2 = box.xyxy[0]
                        center_x = int((x1 + x2) / 2)
                        center_y = int((y1 + y2) / 2)

                        class_id = int(box.cls[0])
                        class_name = model.names[class_id]

                        print(f"객체 감지: {class_name} 위치 ({center_x}, {center_y})")

                        # 메시지 전송
                        msg = f"FIRE@{center_x}@{center_y}@1"
                        length = struct.pack("!I", len(msg))

                        try:
                            sock.sendall(length + msg.encode("utf-8"))
                        except socket.error as e:
                            print(f"전송 오류: {e}")
                            sock.close()
                            exit(1)

                        # 감지된 객체에 원 그리기
#                        cv2.circle(annotated_frame, (center_x, center_y), 5, (0, 255, 0), -1)

            except Exception as e:
                print(f"결과 그리기 오류: {e}")

        # 화면에 실시간 YOLO 감지 결과 출력
        cv2.imshow("YOLOv8", annotated_frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()
    sock.close()


# 메인
if __name__ == "__main__":
#    if len(sys.argv) != 4:
#        print(f"사용법: {sys.argv[0]} <IP> <port> <name>")
#        exit(1)

#    ip = sys.argv[1]
#    port = int(sys.argv[2])
#    name = sys.argv[3]

    # 서버 소켓 설정 및 연결
    server_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_sock.connect((SERVER_IP, SERVER_PORT))
    print("Server connected.")

    # 이미지 수신을 위한 소켓 설정
    obj_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    obj_sock.bind(("0.0.0.0", CAM1_PORT))
    print("Waiting for cam1 connection")

    obj_sock.listen(1)
    pi_sock, _ = obj_sock.accept()
    cam1_ready = True
    print("Cam1 connected")

    # 이미지 수신 스레드 시작
    img_thread = Thread(target=receive_img, args=(pi_sock,))
    img_thread.daemon = True
    img_thread.start()

    # 연결 후 객체 감지 시작
    detect_and_send(server_sock)

