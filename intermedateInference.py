from ultralytics import YOLO
import supervision as sv
model = YOLO('models/eleccomponent-2902ver2.pt', task='detect')
# model = YOLO('folder/model_- 26 february 2024 12_22.onnx', task='detect')

# for result in model.predict(source=0, show=True, imgsz=416, stream=True):
#     # print(result)
#     detection = sv.Detections.from_ultralytics(result)
#     labels = [
#         f"{result.names[class_id]}"
#         for xy, _, conf, class_id, _
#         in detection
#     ]
#     print(labels)
for result in model.predict(source='testImg/testImg2.jpg', show=True, imgsz=416, stream=True):
    detection = sv.Detections.from_ultralytics(result)
    for xyxy, masks, conf, class_id, track_id in detection:
        print(class_id)
