import cv2
from ultralytics import YOLO
import supervision as sv

# model = YOLO('model_- 26 february 2024 12_22.pt', task='detect')
model = YOLO('models/eleccomponent-2902ver2.pt', task='detect')

for result in model.predict(source=0, imgsz=416, stream=True):
    detection = sv.Detections.from_ultralytics(result)
    classID = [
        class_id
        for _, _, _, class_id, _
        in detection
    ]
    # detection attr = xyxy, mask, confidence, class_id, track_id(if available)
    labels = [
        f"detection : {result.names[class_id]}"
        for class_id in classID
    ]
    print(classID)
    frame = result.orig_img
    box_annotator = sv.BoxAnnotator(
        thickness=2,
        text_thickness=1,
        text_scale=0.5
    )

    frames = box_annotator.annotate(
        scene=frame,
        detections=detection,
        labels=labels
    )

    cv2.imshow('detection', frames)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
