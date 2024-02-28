from ultralytics import YOLO
import supervision as sv
# model = YOLO('model_- 26 february 2024 12_22.pt', task='detect')
model = YOLO('folder/model_- 26 february 2024 12_22.onnx', task='detect')

for result in model.predict(source=0, show=True, imgsz=416, stream=True):
    detection = sv.Detections.from_ultralytics(result)
    labels = [
        f"detection : {result.names[class_id]}"
        for _, _, _, class_id, _
        in detection
    ]
    print(labels)
