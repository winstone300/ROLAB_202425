import pyrealsense2 as rs
import numpy as np
import cv2
import threading
from ultralytics import YOLO

class RealSenseYOLO:
    def __init__(self, model_path, width=640, height=480, fps=30, skip_frame=20, region_size=5):
        # 화면 및 처리 관련 설정
        self.width = width
        self.height = height
        self.skip_frame = skip_frame  # 몇 프레임마다 YOLO 추론 요청할지 결정
        self.region_size = region_size  # 중앙 영역의 픽셀 범위 (평균 깊이 계산용)
        
        # YOLO 모델 로드
        self.model = YOLO(model_path)

        # RealSense 파이프라인 및 스트림 설정
        self.pipeline = rs.pipeline()
        self.config = rs.config()
        self.config.enable_stream(rs.stream.depth, width, height, rs.format.z16, fps)
        self.config.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps)
        self.pipeline.start(self.config)

        # Align 설정 (컬러 프레임 기준 깊이 정렬)
        self.align = rs.align(rs.stream.color)

        # 스레드 동기화를 위한 Lock
        self.lock = threading.Lock()

        # YOLO 추론 결과와 최신 컬러 프레임 저장 변수
        self.latest_results = None
        self.processing_flag = False  # YOLO 추론 요청 여부
        self.color_frame = None

        # YOLO 추론을 백그라운드에서 실행하는 스레드 시작
        self.yolo_thread = threading.Thread(target=self.yolo_inference_thread, daemon=True)
        self.yolo_thread.start()

    def yolo_inference_thread(self):
        """백그라운드에서 YOLO 추론을 수행하는 함수"""
        while True:
            if self.processing_flag and self.color_frame is not None:
                with self.lock:
                    frame_copy = self.color_frame.copy()  # 최신 컬러 프레임 복사
                    self.processing_flag = False       # 추론 요청 플래그 해제
                # YOLO 추론 실행 (신뢰도 임계값 0.7 적용)
                results = self.model(frame_copy, conf=0.7)
                with self.lock:
                    self.latest_results = results

    def run(self):
        """RealSense 스트리밍, YOLO 추론 및 추가 기능 실행"""
        frame_count = 0  # 전체 프레임 카운트

        try:
            while True:
                # 프레임 획득 및 컬러/깊이 정렬
                frames = self.pipeline.wait_for_frames()
                aligned_frames = self.align.process(frames)
                aligned_depth_frame = aligned_frames.get_depth_frame()
                aligned_color_frame = aligned_frames.get_color_frame()

                if not aligned_depth_frame or not aligned_color_frame:
                    continue

                # numpy 배열로 변환
                depth_image = np.asanyarray(aligned_depth_frame.get_data())
                self.color_frame = np.asanyarray(aligned_color_frame.get_data())

                frame_count += 1
                # 지정한 프레임 주기마다 YOLO 추론 요청
                if frame_count % self.skip_frame == 0:
                    with self.lock:
                        self.processing_flag = True

                # YOLO 추론 결과가 있으면 시각화
                if self.latest_results is not None:
                    with self.lock:
                        results = self.latest_results

                    # YOLO 객체 검출 결과 오버레이 (annotated_frame)
                    annotated_frame = results[0].plot()

                    # 추가 기능 1: 중앙 선 그리기
                    center_x = self.width // 2
                    cv2.line(annotated_frame, (center_x, 0), (center_x, self.height), (0, 0, 0), 2)

                    # 추가 기능 2: 타겟 박스(녹색) 그리기 (예시: 이미지 중앙에 100x100 크기 박스)
                    target_box_size = 100
                    target_x1 = center_x - target_box_size // 2
                    target_y1 = self.height // 2 - target_box_size // 2
                    target_x2 = center_x + target_box_size // 2
                    target_y2 = self.height // 2 + target_box_size // 2
                    cv2.rectangle(annotated_frame, (target_x1, target_y1), (target_x2, target_y2), (0, 255, 0), 3)

                    # 추가 기능 3: 중앙 영역의 깊이 값 평균 계산 및 표시
                    center_y = self.height // 2
                    x1 = max(0, center_x - self.region_size)
                    x2 = min(self.width, center_x + self.region_size)
                    y1 = max(0, center_y - self.region_size)
                    y2 = min(self.height, center_y + self.region_size)
                    center_region = depth_image[y1:y2, x1:x2]
                    if center_region.size > 0:
                        avg_depth = np.mean(center_region)
                        cv2.putText(annotated_frame, f"Depth: {avg_depth:.2f}", (10, 30),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

                    # 깊이 데이터를 컬러맵으로 변환 (오른쪽에 표시)
                    depth_colormap = cv2.applyColorMap(
                        cv2.convertScaleAbs(depth_image, alpha=0.03),
                        cv2.COLORMAP_JET
                    )
                    # 컬러 이미지와 깊이 맵을 나란히 결합하여 표시
                    stacked_frame = np.hstack((annotated_frame, depth_colormap))
                    cv2.imshow("RealSense + YOLO (Color / Depth)", stacked_frame)

                # 'q' 키를 누르면 종료
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

        finally:
            self.cleanup()

    def cleanup(self):
        self.pipeline.stop()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    # 클래스 기반 시스템 실행 (두 번째 코드의 기능 모두 포함)
    system = RealSenseYOLO("/home/rolab/test_code/ROLAB_202425/best.pt")
    system.run()
