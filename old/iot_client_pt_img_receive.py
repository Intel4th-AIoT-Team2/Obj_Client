import cv2
from sympy import im
from ultralytics import YOLO
import socket
import sys
from queue import Queue
import numpy as np
from threading import Thread

# YOLO 모델 초기화
model = YOLO('fire.pt')

# 신뢰도 임계값 (80%)
CONF_THRESHOLD = 0.80

# 이미지 프레임을 위한 큐
img_queue = Queue()

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
    # 이미지 데이터는 다음과 같은 방식으로 전송됨:
    # client_socket.sendall((str(len(byteData))).encode().ljust(16) + byteData)
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
            img = cv2.imdecode(byte_data, cv2.IMREAD_COLOR)
            img_queue.put(img)
        except Exception as e:
            print(f"이미지 수신 오류: {e}")
            break
    sock.close()


# 객체 감지 및 좌표 전송
def detect_and_send(sock, name):
    while True:
        # 큐에서 이미지 가져오기
        if img_queue.empty():
            continue
        frame = img_queue.get()

        # YOLO 모델을 사용해 객체 감지
        results = model(frame)
        annotated_frame = frame

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
                        msg = f"[ALLMSG]{class_name}@{center_x}@{center_y}\n"
                        try:
                            sock.sendall(msg.encode())
                        except socket.error as e:
                            print(f"전송 오류: {e}")
                            sock.close()
                            exit(1)

                        # 감지된 객체에 원 그리기
                        cv2.circle(annotated_frame, (center_x, center_y), 5, (0, 255, 0), -1)

            except Exception as e:
                print(f"결과 그리기 오류: {e}")

        # 화면에 실시간 YOLO 감지 결과 출력
        cv2.imshow("YOLOv8 실시간 감지", annotated_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()
    sock.close()

# 메인 함수
def main():
    if len(sys.argv) != 4:
        print(f"사용법: {sys.argv[0]} <IP> <port> <name>")
        exit(1)

    ip = sys.argv[1]
    port = int(sys.argv[2])
    name = sys.argv[3]

    # 소켓 설정 및 연결
    sock = setup_socket(ip, port, name)

    # 이미지 수신을 위한 소켓 설정
    obj_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    obj_sock.bind(("0.0.0.0", 5001))
    print("연결 대기 중")
    obj_sock.listen(1)
    pi_sock, _ = obj_sock.accept()
    print("연결 완료")

    # pi_sock에 연결 메시지 전송
    pi_sock.sendall("서버와 연결되었습니다.".encode("utf-8"))

    # 이미지 수신 스레드 시작
    img_thread = Thread(target=receive_img, args=(pi_sock,))
    img_thread.daemon = True
    img_thread.start()

    # 연결 후 객체 감지 시작
    detect_and_send(sock, name)

if __name__ == "__main__":
    main()
