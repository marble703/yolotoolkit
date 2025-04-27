from ultralytics import YOLO

# Load the YOLO11 model
model = YOLO("pts/best_0426_noamp.pt")

# Export the model to ONNX format
model.export(format="onnx")  # creates 'yolov8n.onnx'