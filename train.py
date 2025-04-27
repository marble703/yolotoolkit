from ultralytics import YOLO

# Load a COCO-pretrained YOLOv8n model
model = YOLO("yolov8l.pt")

# Display model information (optional)
model.info()

# Train the model on the merged dataset for 100 epochs
results = model.train(
    data="/home/chen/arm/yolo/my_origindata/Augumentation/splited/data.yaml",
    epochs=100,
    imgsz=640,
    batch=0.8,
    device=0,
    amp=False  # 禁用混合精度训练
)