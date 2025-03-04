import pyrealsense2 as rs
import numpy as np
import cv2
import threading
from ultralytics import YOLO

class RealSenseYoloDetector:
    def __init__(self, yolo_model_path="best.pt", conf_threshold=0.7):
        """초기 설정: YOLO 모델, RealSense 파이프라인, Align 등."""
        self.model = YOLO(yolo_model_path)
        self.conf_threshold = conf_threshold

        # RealSense 초기화
        self.pipeline = rs.pipeline()
        self.config = rs.config()
        self.config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
        self.config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        self.pipeline.start(self.config)

        self.align_to = rs.stream.color
        self.align = rs.align(self.align_to)

        # 화면 해상도 및 타겟 박스 정보
        self.width, self.height = 640, 480
        self.center_x, self.center_y = self.width // 2, self.height // 2
        self.target_x1, self.target_y1 = self.center_x - 50, self.height - 200
        self.target_x2, self.target_y2 = self.center_x + 50, self.height

        # 추론 결과 저장용
        self.best_pet_box = None
        self.highest_conf = 0.0
        self.avg_dist = 0.0
        self.state = ""

        # 33번째 프레임마다 YOLO 추론
        self.frame_counter = 0

        # 거리 계산 시 사용되는 범위 설정
        self.region_size = 5
    
    def detect_pet_bottle(self, color_image):
        """
        YOLO 추론을 수행하여 'PET Bottle' 중
        가장 높은 신뢰도의 박스를 self.best_pet_box에 갱신.
        """
        results = self.model(color_image, conf=self.conf_threshold)
        # 첫 번째 결과에 대한 기본 시각화(annotated_frame)는 run()에서 처리
        self.highest_conf = 0.0
        self.best_pet_box = None

        for result in results:
            boxes = result.boxes
            for box in boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cls_id = int(box.cls[0])
                class_name = self.model.names[cls_id]
                conf = float(box.conf[0])

                if class_name == "PET Bottle":
                    if conf > self.highest_conf:
                        self.highest_conf = conf
                        self.best_pet_box = box
    
    def run(self):
        """메인 루프: 매 프레임 수신, 33번째마다 YOLO 추론 쓰레드 실행."""
        try:
            while True:
                self.frame_counter += 1

                # 1) 프레임 획득 및 컬러/깊이 정렬
                frames = self.pipeline.wait_for_frames()
                aligned_frames = self.align.process(frames)

                aligned_depth_frame = aligned_frames.get_depth_frame()
                aligned_color_frame = aligned_frames.get_color_frame()

                if not aligned_depth_frame or not aligned_color_frame:
                    continue

                # numpy 배열 변환
                depth_image = np.asanyarray(aligned_depth_frame.get_data())
                color_image = np.asanyarray(aligned_color_frame.get_data())

                # (a) 33번째 프레임마다 YOLO 추론 (동기화 - 다른 코드 기다림)
                if self.frame_counter % 33 == 0:
                    # 스레드 생성 -> 시작 -> 종료까지 대기
                    detect_thread = threading.Thread(
                        target=self.detect_pet_bottle, 
                        args=(color_image,)
                    )
                    detect_thread.start()
                    detect_thread.join()

                # (b) YOLO 시각화(기본) - last inference result
                #    Ultralytics는 한 번 추론 시 results[0].plot()을 이용하지만
                #    여기서는 self.best_pet_box와 별도로, 중간에 model(...) 생략
                #    -> 화면 표시용으로 dummy inference or blank
                #    -> 실제로는 매 프레임마다 plot() 할 필요가 없으니 간단히 처리
                # ---------------------------------------------------------
                # 간단히: "최근" color_image에 대한 plot만 (추론 안 해도 됨)
                #        다만, 계속해서 model(...).plot()을 호출하면 속도↓
                # ---------------------------------------------------------
                # 여기서는 33번째 프레임에만 실제 추론했으니, 그때의 result만 써야 함.
                # -> 간단히 "annotated_frame = color_image.copy()"로 대체
                annotated_frame = color_image.copy()

                # -------------------------
                # [아래는 기존 시각화 및 로직]
                # -------------------------
                # 1) 상단 검정 바, 중심선, 타겟 박스
                cv2.rectangle(annotated_frame, (0, 0), (self.width, 30), (0, 0, 0), -1)
                cv2.line(annotated_frame, (0, self.center_y), (self.width, self.center_y), (0, 0, 0), 2)
                cv2.line(annotated_frame, (self.center_x, 0), (self.center_x, self.height), (0, 0, 0), 2)
                cv2.rectangle(annotated_frame, (self.target_x1, self.target_y1),
                              (self.target_x2, self.target_y2), (0, 255, 0), 3)

                depth_colormap = cv2.applyColorMap(
                    cv2.convertScaleAbs(depth_image, alpha=0.03),
                    cv2.COLORMAP_JET
                )

                self.state = ""
                if self.best_pet_box is not None:
                    bx1, by1, bx2, by2 = map(int, self.best_pet_box.xyxy[0])
                    bx, by = (bx1 + bx2) / 2, (by1 + by2) / 2

                    # (2) 컬러 프레임 표시 (빨간 박스 + 신뢰도)
                    cv2.rectangle(annotated_frame, (bx1, by1), (bx2, by2), (0, 0, 255), 3)
                    cv2.putText(
                        annotated_frame,
                        f"Highest PET: {self.highest_conf:.2f}",
                        (bx1, max(by1 - 10, 0)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                        (0, 0, 255), 2
                    )

                    # 로봇 유도 상태
                    if bx <= self.target_x1:
                        self.state = "Move Left"
                    elif bx >= self.target_x2:
                        self.state = "Move Right"
                    else:
                        if by <= self.target_y1:
                            self.state = "Move Forward"
                        else:
                            self.state = "In Target Box"

                    cv2.putText(annotated_frame, f"pos: ({int(bx)}, {int(by)})", (0, 25),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                    cv2.putText(annotated_frame, f"state: {self.state}", (self.center_x, 25),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

                    # (3) 깊이 영상 표시 (박스 + min 거리)
                    cv2.rectangle(depth_colormap, (bx1, by1), (bx2, by2), (0, 0, 255), 3)

                    # min 거리 계산
                    cx, cy = int(bx), int(by)
                    x_region_size = int((max(bx1, bx2) - bx) / 2)
                    y_region_size = int((max(by1, by2) - by) / 2)
                    start_x = max(cx - x_region_size, 0)
                    end_x = min(cx + x_region_size, self.width - 1)
                    start_y = max(cy - y_region_size, 0)
                    end_y = min(cy + y_region_size, self.height - 1)

                    depth_values = []
                    min_val = 1e9
                    for px in range(start_x, end_x + 1):
                        for py in range(start_y, end_y + 1):
                            dist_val = aligned_depth_frame.get_distance(px, py)
                            if dist_val > 0:
                                min_val = min(min_val, dist_val)

                    if min_val < 1e9:
                        self.avg_dist = min_val
                    else:
                        self.avg_dist = 0.0

                    cv2.putText(depth_colormap,
                                f"Dist: {self.avg_dist:.2f}m",
                                (bx1, max(by1 - 10, 0)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                                (0, 0, 255), 2)
                else:
                    # PET Bottle이 인식되지 않은 경우
                    cv2.putText(annotated_frame, "No Object", (0, 25),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

                # 컬러+깊이 합쳐서 표시
                stacked_frame = np.hstack((annotated_frame, depth_colormap))
                cv2.imshow("RealSense + YOLO (Color / Depth)", stacked_frame)

                # 종료 조건
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

        finally:
            self.pipeline.stop()
            cv2.destroyAllWindows()

def main():
    detector = RealSenseYoloDetector(
        yolo_model_path="/home/rolab/test_code/ROLAB_202425/best.pt",
        conf_threshold=0.7
    )
    detector.run()

if __name__ == "__main__":
    main()
