from ultralytics import YOLO

model = YOLO('model_- 26 february 2024 12_22.pt', task='detect')

model.export(format='onnx', imgsz=(416, 416))
