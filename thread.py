import cv2
import threading
import queue
import time
import numpy as np
from ultralytics import YOLO
import pyrealsense2 as rs

# ------------------------
# 1. 모델 및 RealSense 초기화
# ------------------------

# YOLO 모델 로드
model = YOLO("/home/rolab/test_code/ROLAB_202425/best.pt")  # 모델 파일 경로 수정 가능

# RealSense 파이프라인 및 스트림 설정
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
profile = pipeline.start(config)

# 컬러 프레임과 깊이 프레임 정렬 객체 생성 (컬러 기준)
align = rs.align(rs.stream.color)

# 프레임을 안전하게 공유하기 위한 큐 (최대 10개까지 저장)
frame_queue = queue.Queue(maxsize=10)

# 전역 종료 플래그
exit_flag = threading.Event()

# ------------------------
# 2. 스레드 함수 정의
# ------------------------

def capture_frames():
    """
    RealSense 카메라로부터 프레임을 캡처하여 큐에 저장하는 스레드 함수.
    매 20번째 프레임만 큐에 넣어 (연산 부하를 줄이기 위해) 처리한다.
    """
    cnt = 0
    while not exit_flag.is_set():
        try:
            frames = pipeline.wait_for_frames()
            aligned_frames = align.process(frames)
            cnt += 1

            # 매 20번째 프레임마다 처리 (프레임 건너뛰기)
            if cnt % 20 != 0:
                continue
            cnt = 0

            # 컬러 및 깊이 프레임 획득
            aligned_color_frame = aligned_frames.get_color_frame()
            aligned_depth_frame = aligned_frames.get_depth_frame()
            if not aligned_color_frame or not aligned_depth_frame:
                continue

            # 컬러 프레임을 numpy 배열로 변환
            color_image = np.asanyarray(aligned_color_frame.get_data())

            # 큐가 가득 찼다면 오래된 프레임 삭제
            if frame_queue.full():
                try:
                    frame_queue.get_nowait()
                except queue.Empty:
                    pass

            frame_queue.put(color_image)
        except Exception as e:
            print("캡처 스레드 오류:", e)
            break

def process_frames():
    """
    큐에서 프레임을 가져와 YOLO 모델로 추론한 후 결과를 화면에 표시하는 스레드 함수.
    'q' 키를 누르면 종료한다.
    """
    while not exit_flag.is_set():
        try:
            if not frame_queue.empty():
                color_image = frame_queue.get()
                # YOLO 모델로 추론 수행
                results = model(color_image)
                # 결과를 시각화 (annotated_frame에 추론 결과가 그림으로 표시됨)
                annotated_frame = results[0].plot()
                
                cv2.imshow("YOLO Detection", annotated_frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    exit_flag.set()
            else:
                time.sleep(0.01)
        except Exception as e:
            print("처리 스레드 오류:", e)
            break

# ------------------------
# 3. 스레드 시작 및 종료 처리
# ------------------------

# 프레임 캡처 스레드 생성 및 시작
capture_thread = threading.Thread(target=capture_frames)
capture_thread.start()

# 프레임 처리 스레드 생성 및 시작
process_thread = threading.Thread(target=process_frames)
process_thread.start()

try:
    # 메인 스레드에서는 두 스레드가 종료될 때까지 대기
    while not exit_flag.is_set():
        time.sleep(0.1)
except KeyboardInterrupt:
    exit_flag.set()
finally:
    # 종료 플래그 설정 후 스레드 종료 대기
    capture_thread.join()
    process_thread.join()
    pipeline.stop()
    cv2.destroyAllWindows()
