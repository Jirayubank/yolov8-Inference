from ultralytics import YOLO

model = YOLO('elecCompo-2902Ver1.pt', task='detect')
# dataStreaming = 'rtsp://admin:tatc1234@192.168.1.64:554/Streaming/Channels/101'

model.predict(source='testimg45jpg.jpg', show=True, save=True, imgsz=640)

