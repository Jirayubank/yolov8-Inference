from ultralytics import YOLO

model = YOLO('models/eleccomponent-2902ver2.pt', task='detect')
# dataStreaming = 'rtsp://admin:tatc1234@192.168.1.64:554/Streaming/Channels/101'

for result in model.predict(
    source='testImg/testimg1.jpg',
    conf=0.5,
    show=True,
    save=True,
    classes=[0, 1],
    imgsz=(640, 640),
    agnostic_nms=True
):
    print(result)
