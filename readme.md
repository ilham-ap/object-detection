# Google Colab: Access Webcam for Realtime Object Detection
<a href="https://colab.research.google.com/github/ilham-ap/object-detection/blob/main/object_detection_colab_webcam.ipynb" target="parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/><br/>

This notebook will go through how to access and run code object detection using your webcam.<br/>
For this purpose of this tutorial we will be using ultralytics YOLO to do object detection on our Webcam.

# Google Colab: Video Object Detection
<a href="https://colab.research.google.com/github/ilham-ap/object-detection/blob/main/video_object_detection.ipynb" target="parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/><br/>

#### Vehicles Detection, Tracking and Counting 
![](./img/ezgif-3-fbb5a4d769.gif)

# Realtime vehicle detection with IP WEBCAM android
#### 1. download IP WEBCAM and Termux on android 
#### 2. download ngrok linux https://ngrok.com/download
#### 3. run ngrok, maksure folder of ngrok for run, if in downloads
```
./downloads/ngrok tcp 8080
```
#### 4. put your ngrok addres and port, example
```
from ultralytics import YOLO
import cv2
model = YOLO('yolov8n.pt')  # Load an official Detect model
results = model.track(source='rtsp://2.tcp.ngrok.io:16976/h264.sdp', show=True)  # Tracking with default tracker
```
#### 5. open ip webcam on android and run script on your terminal <br/>
<iframe width="560" height="315" src="https://www.youtube.com/embed/nQ-eMwcqYIQ" frameborder="0" allowfullscreen></iframe>


