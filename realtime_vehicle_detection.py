from ultralytics import YOLO
model = YOLO('yolov8n.pt')

result = model.track(source='rtsp://2.tcp.ngrok.io:16976/h264.sdp', show=True)