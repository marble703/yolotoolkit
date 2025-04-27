import cv2
import time
from ultralytics import YOLO

# 加载 YOLOv8 模型
model = YOLO("./pts/best0426_4.pt")

# 打开默认摄像头
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise IOError("无法打开摄像头")

prev_frame_time = 0
new_frame_time = 0

while(True):
    ret, frame = cap.read()
    if not ret:
        break

    new_frame_time = time.time()

    # 使用 YOLOv8 模型进行推理
    model.to('cpu')
    # model.to('cuda:0')  # 将模型移动到 GPU
    results = model.predict(frame, conf=0.25, device='cuda:0')  # 使用 GPU 进行推理
    # results = model(frame)

    # 在图像上绘制检测结果
    for result in results:
        boxes = result.boxes
        # print(result)
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
            confidence = box.conf[0].cpu().numpy()
            class_id = int(box.cls[0].cpu().numpy())
            class_name = model.names[class_id]

            label = f"{class_name} {confidence:.2f}"
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    end_time = time.time()

    # 计算帧率
    fps = 1/(new_frame_time-prev_frame_time)
    prev_frame_time = new_frame_time
    fps = int(fps)
    fps_text = str(fps)

    frame_time = end_time - new_frame_time

    # 显示帧率
    cv2.putText(frame, fps_text, (7, 70), cv2.FONT_HERSHEY_SIMPLEX, 3, (100, 255, 0), 3, cv2.LINE_AA)
    cv2.putText(frame, f"Frame Time: {frame_time:.2f}s", (7, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # 显示结果
    cv2.imshow('YOLOv8 Detection', frame)

    # 按下 'q' 键退出
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 释放摄像头和窗口
cap.release()
cv2.destroyAllWindows()