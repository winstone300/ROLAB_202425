import cv2
import threading
import queue
import time
import numpy as np
from ultralytics import YOLO
import pyrealsense2 as rs

class RealSenseYOLO:
    def __init__(self, model_path, width=640, height=480, fps=30, frame_skip=20, queue_size=10):
        """
        초기화: YOLO 모델 로드, RealSense 카메라 및 스트림 설정, 큐와 스레드 준비
        """
        self.model_path = model_path
        self.width = width
        self.height = height
        self.fps = fps
        self.frame_skip = frame_skip  # 몇 프레임마다 처리할지 결정

        # YOLO 모델 로드
        self.model = YOLO(model_path)

        # RealSense 파이프라인 및 스트림 설정
        self.pipeline = rs.pipeline()
        self.config = rs.config()
        self.config.enable_stream(rs.stream.color, self.width, self.height, rs.format.bgr8, self.fps)
        self.config.enable_stream(rs.stream.depth, self.width, self.height, rs.format.z16, self.fps)
        self.pipeline.start(self.config)

        # 컬러 프레임 기준으로 깊이 프레임 정렬
        self.align = rs.align(rs.stream.color)

        # 프레임을 안전하게 공유하기 위한 큐
        self.frame_queue = queue.Queue(maxsize=queue_size)

        # 스레드 종료 플래그
        self.exit_flag = threading.Event()

        # 스레드 초기화 (아직 시작하지 않음)
        self.capture_thread = threading.Thread(target=self.capture_frames)
        self.process_thread = threading.Thread(target=self.process_frames)

    def capture_frames(self):
        """
        RealSense 카메라로부터 프레임을 캡처하여 큐에 저장하는 스레드 함수.
        매 frame_skip번째 프레임만 큐에 넣어 연산 부하를 줄입니다.
        """
        cnt = 0
        while not self.exit_flag.is_set():
            try:
                frames = self.pipeline.wait_for_frames()
                aligned_frames = self.align.process(frames)
                cnt += 1
                if cnt % self.frame_skip != 0:
                    continue
                cnt = 0

                # 컬러 및 깊이 프레임 획득
                aligned_color_frame = aligned_frames.get_color_frame()
                aligned_depth_frame = aligned_frames.get_depth_frame()
                if not aligned_color_frame or not aligned_depth_frame:
                    continue

                # 컬러 프레임을 numpy 배열로 변환
                color_image = np.asanyarray(aligned_color_frame.get_data())

                # 큐가 가득 찼으면 오래된 프레임 삭제
                if self.frame_queue.full():
                    try:
                        self.frame_queue.get_nowait()
                    except queue.Empty:
                        pass

                self.frame_queue.put(color_image)
            except Exception as e:
                print("캡처 스레드 오류:", e)
                break

    def process_frames(self):
        """
        큐에서 프레임을 가져와 YOLO 모델로 추론 후 결과를 화면에 표시하는 스레드 함수.
        'q' 키를 누르면 종료합니다.
        """
        while not self.exit_flag.is_set():
            try:
                if not self.frame_queue.empty():
                    color_image = self.frame_queue.get()
                    # YOLO 모델로 추론 수행
                    results = self.model(color_image)
                    # 결과 시각화: 추론 결과가 그림으로 표시된 이미지 반환
                    annotated_frame = results[0].plot()

                    cv2.imshow("YOLO Detection", annotated_frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        self.exit_flag.set()
                else:
                    time.sleep(0.01)
            except Exception as e:
                print("처리 스레드 오류:", e)
                break

    def start(self):
        """
        캡처 및 처리 스레드 실행
        """
        self.capture_thread.start()
        self.process_thread.start()

    def stop(self):
        """
        종료 플래그를 설정하고, 스레드 종료 대기 및 자원 해제
        """
        self.exit_flag.set()
        self.capture_thread.join()
        self.process_thread.join()
        self.pipeline.stop()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    model_path = "/home/rolab/test_code/ROLAB_202425/best.pt"
    app = RealSenseYOLO(model_path)
    app.start()

    try:
        # 메인 스레드는 exit_flag가 설정될 때까지 대기
        while not app.exit_flag.is_set():
            time.sleep(0.1)
    except KeyboardInterrupt:
        app.stop()
    finally:
        app.stop()
