# Obj_Client
- Python 파일 설명
  - iot_client_pt_.py : 연결된 웹캠에서 받은 영상으로 Object Detection을 진행해 탐지된 객체의 중앙 좌표 값을 iot server에 전송
  - iot_client_pt_img_receive.py : iot server에 연결 된 CCTV에서 받은 영상으로 Object Detection을 진행해 탐지된 객체의 중앙 좌표 값을 iot server에 전송
- YOLO 모델 설명
  - fire_v1 : 불꽃 관련 데이터를 100번 학습한 데이터
  - fire_v2 : 고체연료에 붙은 불꽃 관련 데이터 셋 3000장을 100번 학습한 데이터
  - fire_v3 : 고체연료에 붙은 불꽃 관련 데이터 셋을 1000장으로 줄여 100번 학습한 데이터
  - fire : 고체연료에 붙은 불꽃 관련 데이터 셋 1000장에 추가로 CCTV에서 촬영한 고체연료 불꽃 데이터 셋을 추가해 100번 학습한 데이터
- 실행 방법
  - python3 iot_client_pt.py IP PORT NAME
  - python3 iot_client_pt_img_receive.py IP PORT NAME
  - IP : 연결할 서버의 IP 작성
  - PORT : 서버의 PORT 작성
  - NAME : 서버에 등록된 NAME 작성
