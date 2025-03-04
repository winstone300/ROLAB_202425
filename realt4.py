import pyrealsense2 as rs
import numpy as np
import cv2
import threading
from ultralytics import YOLO

class RealSenseYOLO:
    def __init__(self, model_path):
        # YOLO 모델 로드
        self.model = YOLO(model_path)

        # RealSense 파이프라인 설정
        self.pipeline = rs.pipeline()
        self.config = rs.config()
        self.config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
        self.config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        self.pipeline.start(self.config)

        # Align 설정 (컬러 프레임 기준으로 깊이 정렬)
        self.align = rs.align(rs.stream.color)

        # 스레드 동기화를 위한 Lock
        self.lock = threading.Lock()

        # YOLO 추론 결과 저장 변수
        self.latest_results = None
        self.processing_flag = False  # YOLO 실행 여부

        # YOLO 백그라운드 스레드 실행
        self.yolo_thread = threading.Thread(target=self.yolo_inference_thread, daemon=True)
        self.yolo_thread.start()

    def yolo_inference_thread(self):
        """ YOLO 추론을 백그라운드에서 실행하는 함수 """
        while True:
            if self.processing_flag:
                with self.lock:
                    frame_copy = self.color_frame.copy()  # 최신 프레임 복사
                    self.processing_flag = False  # 처리 중 상태 해제

                # YOLO 추론 실행
                results = self.model(frame_copy, conf=0.7)

                # YOLO 결과 업데이트
                with self.lock:
                    self.latest_results = results

    def run(self):
        """ RealSense 스트리밍 & YOLO 추론 실행 """
        frame_count = 0  # 프레임 카운트

        try:
            while True:
                # 프레임 획득 및 컬러/깊이 정렬
                frames = self.pipeline.wait_for_frames()
                aligned_frames = self.align.process(frames)
                aligned_depth_frame = aligned_frames.get_depth_frame()
                aligned_color_frame = aligned_frames.get_color_frame()

                if not aligned_depth_frame or not aligned_color_frame:
                    continue

                # numpy 배열 변환
                depth_image = np.asanyarray(aligned_depth_frame.get_data())
                self.color_frame = np.asanyarray(aligned_color_frame.get_data())

                # YOLO 추론 요청 (33 프레임마다 실행)
                if frame_count % 33 == 0:
                    with self.lock:
                        self.processing_flag = True  # YOLO가 새로운 프레임을 처리하도록 설정
                frame_count += 1

                # YOLO 결과 시각화
                if self.latest_results is not None:
                    with self.lock:
                        results = self.latest_results  # 최신 YOLO 결과 가져오기

                    # YOLO 시각화
                    annotated_frame = results[0].plot()

                    # 깊이 맵 생성 (오른쪽 표시)
                    depth_colormap = cv2.applyColorMap(
                        cv2.convertScaleAbs(depth_image, alpha=0.03),
                        cv2.COLORMAP_JET
                    )
                    stacked_frame = np.hstack((annotated_frame, depth_colormap))
                    cv2.imshow("RealSense + YOLO (Color / Depth)", stacked_frame)

                # q 키로 종료
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

        finally:
            self.pipeline.stop()
            cv2.destroyAllWindows()

if __name__ == "__main__":
    # 실행
    yolo_system = RealSenseYOLO("/home/rolab/test_code/ROLAB_202425/best.pt")
    yolo_system.run()
