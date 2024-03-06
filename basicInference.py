from ultralytics import YOLO

model = YOLO('models/elecCompo-2902Ver1.pt', task='detect')
# model = YOLO(args.model, task='detect')

result = model.predict(source='testImg/testImg1.jpg', show=True, save=True, conf=0.6, agnostic_nms=True, imgsz=(640, 640),
                       stream=True)
for results in result:
    print(results)