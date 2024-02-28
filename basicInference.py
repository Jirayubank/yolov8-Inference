from ultralytics import YOLO

model = YOLO('folder/model_- 26 february 2024 12_22.onnx', task='detect')
dataStreaming = 'rtsp://admin:tatc1234@192.168.1.64:554/Streaming/Channels/101'
model.predict(source=dataStreaming, show=True, imgsz=416)
