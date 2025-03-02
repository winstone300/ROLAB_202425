from ultralytics import YOLO
import cv2

CONF = 0.8  # 신뢰도(confidence) 설정
CAM_NUM = 0 # 캠 번호 설정(0:노트북, 2:웹캠)

# Load a model
model = YOLO("best.pt")

# Open the camera
webcam = cv2.VideoCapture(CAM_NUM)

# Check if the camera opened successfully
if not webcam.isOpened():
    print("웹캠을 열 수 없습니다. 프로그램을 종료합니다.")
    exit()

# Define colors for different labels
colors = {
    'bottle' : (0, 0, 255),   # red
}

while True:
    ret, frame = webcam.read()
    if not ret:
        print("카메라에서 프레임을 읽을 수 없습니다. 프로그램을 종료합니다.")
        break
    
    # Run inference on the frame
    results = model(frame, imgsz=640, conf=CONF)  # 이미지 크기와 신뢰도(confidence)를 설정

    # Display the results
    
    for result in results:
        boxes = result.boxes
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
            conf = box.conf[0]
            cls = box.cls[0]
            label = model.names[int(cls)]

            # Choose color based on label
            color = colors.get(label, (255, 255, 255))

            # Draw the box
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)  # 바운딩 박스 그리기
            cv2.putText(frame, f'{label}: {conf:.2f}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)  # 라벨 및 신뢰도 표시
    

    # 프레임 출력
    cv2.imshow("Webcam", frame)

    # 'q' 키를 누르면 종료
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 자원 해제
webcam.release()
cv2.destroyAllWindows()