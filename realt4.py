import pyrealsense2 as rs
import numpy as np
import cv2
from ultralytics import YOLO

# 1. YOLO 모델 로드
model = YOLO("best.pt")  # 모델 파일 경로 수정 가능

# 2. RealSense 파이프라인 및 스트림 설정
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

# 3. 파이프라인 시작
pipeline.start(config)

# 4. 컬러 스트림에 맞춰 깊이 정렬(Align) 설정
align_to = rs.stream.color
align = rs.align(align_to)

try:
    # 화면 및 타겟 범위 정보
    width, height = 640, 480
    center_x, center_y = width // 2, height // 2
    target_x1, target_y1 = center_x - 50, height - 200
    target_x2, target_y2 = center_x + 50, height

    # 중앙 주변 픽셀을 평균낼 때 사용할 범위
    region_size = 5  # 예: 중심 좌표 주변 (±5) 픽셀
    
    while True:
        # 5. 프레임 획득 및 컬러/깊이 정렬
        frames = pipeline.wait_for_frames()
        aligned_frames = align.process(frames)

        aligned_depth_frame = aligned_frames.get_depth_frame()
        aligned_color_frame = aligned_frames.get_color_frame()

        if not aligned_depth_frame or not aligned_color_frame:
            continue

        # numpy 배열 변환 (정렬된 컬러/깊이)
        depth_image = np.asanyarray(aligned_depth_frame.get_data())
        color_image = np.asanyarray(aligned_color_frame.get_data())

        # 6. YOLO 추론
        results = model(color_image, conf=0.7)
        annotated_frame = results[0].plot()  # YOLO 기본 시각화

        highest_conf = 0.0
        best_pet_box = None

        # -------------------------
        # 7. 여러 바운딩박스 중 'PET Bottle'에 해당하는 것 중
        #    신뢰도(Confidence)가 가장 높은 박스 찾기
        # -------------------------
        for result in results:
            boxes = result.boxes
            for box in boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cls_id = int(box.cls[0])
                class_name = model.names[cls_id]
                conf = float(box.conf[0])

                if class_name == "PET Bottle":  # 모델 라벨명에 맞게 수정
                    if conf > highest_conf:
                        highest_conf = conf
                        best_pet_box = box

        # -------------------------
        # 8. 화면 안내선(텍스트 박스, 중앙선, 타겟 박스 등)
        # -------------------------
        # 상단 검정 바
        cv2.rectangle(annotated_frame, (0, 0), (width, 30), (0, 0, 0), -1)
        # 중심선 (x, y축)
        cv2.line(annotated_frame, (0, center_y), (width, center_y), (0, 0, 0), 2)
        cv2.line(annotated_frame, (center_x, 0), (center_x, height), (0, 0, 0), 2)
        # 타겟 박스(녹색)
        cv2.rectangle(annotated_frame, (target_x1, target_y1),
                      (target_x2, target_y2), (0, 255, 0), 3)

        # 깊이 데이터를 컬러맵으로 변환 (오른쪽에 표시)
        depth_colormap = cv2.applyColorMap(
            cv2.convertScaleAbs(depth_image, alpha=0.03),
            cv2.COLORMAP_JET
        )

        state = ""
        if best_pet_box is not None:
            bx1, by1, bx2, by2 = map(int, best_pet_box.xyxy[0])
            bx, by = (bx1 + bx2) / 2, (by1 + by2) / 2

            # -------------------------
            # (1) 컬러 영상에 PET 박스 표시 (빨간 박스)
            # -------------------------
            cv2.rectangle(annotated_frame, (bx1, by1), (bx2, by2), (0, 0, 255), 3)
            cv2.putText(
                annotated_frame,
                f"Highest PET: {highest_conf:.2f}",
                (bx1, max(by1 - 10, 0)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                (0, 0, 255), 2
            )

            # 로봇 유도 상태 결정 (타겟 박스 안으로)
            if bx <= target_x1:
                state = "Move Left"
            elif bx >= target_x2:
                state = "Move Right"
            else:
                if by <= target_y1:
                    state = "Move Forward"
                else:
                    state = "In Target Box"

            cv2.putText(annotated_frame, f"pos: ({int(bx)}, {int(by)})", (0, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(annotated_frame, f"state: {state}", (center_x, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            # -------------------------
            # (2) 깊이 영상에도 동일 박스 + 거리 표시
            # -------------------------
            cv2.rectangle(depth_colormap, (bx1, by1), (bx2, by2), (0, 0, 255), 3)

            # 중앙 주변 (region_size × region_size) 픽셀에서 유효 깊이 평균
            cx, cy = int(bx), int(by)
            x_region_size = int((max(bx1,bx2)-bx)/2)
            y_region_size = int((max(by1,by2)-by)/2)
            start_x = max(cx - x_region_size, 0)
            end_x = min(cx + x_region_size, width - 1)
            start_y = max(cy - y_region_size, 0)
            end_y = min(cy + y_region_size, height - 1)

            depth_values = []
            min_val=1e9
            for px in range(start_x, end_x + 1):
                for py in range(start_y, end_y + 1):
                    dist_val = aligned_depth_frame.get_distance(px, py)
                    if dist_val > 0:  # 유효 깊이만 수집
                        depth_values.append(dist_val)
                        min_val= min(dist_val,min_val)

            if len(depth_values) > 0:
                avg_dist = min_val
            else:
                avg_dist = 0.0  # 주변 픽셀 모두 0이면 측정 불가

            cv2.putText(depth_colormap,
                        f"Dist: {avg_dist:.2f}m",
                        (bx1, max(by1 - 10, 0)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                        (0, 0, 255), 2)
        else:
            # PET Bottle이 아예 인식되지 않은 경우
            cv2.putText(annotated_frame, "No Object", (0, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        # 컬러 영상(왼쪽) + 깊이 컬러맵(오른쪽) 합쳐서 표시
        stacked_frame = np.hstack((annotated_frame, depth_colormap))
        cv2.imshow("RealSense + YOLO (Color / Depth)", stacked_frame)

        # q 키로 종료
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    # 리소스 해제
    pipeline.stop()
    cv2.destroyAllWindows()
