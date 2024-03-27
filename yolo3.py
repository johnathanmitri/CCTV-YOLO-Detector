import torch
from ultralytics import YOLO
from PIL import Image
import cv2
import supervision as sv
import numpy as np
import time
import winsound
from protected.credentials import RTSP_CREDENTIALS

frequency = 1200  # Set Frequency in Hertz
duration = 0  # Set Duration in ms
winsound.Beep(frequency, duration)

#import torch.nn.functional as F

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Using device: {device}')

# Load model
model = YOLO('yolov8n.pt').to(device)

cap = cv2.VideoCapture("rtsp://" + RTSP_CREDENTIALS + "@192.168.254.166:554/axis-media/media.amp")
detections = None

box_annotator = sv.BoxAnnotator(thickness=2, text_thickness=1, text_scale=0.5)

person = 0

i = 0
frameTimes = []

#startTime = time.time()
#frames = []
lastBeep = time.time()
while(cap.isOpened()): # and time.time()-startTime < 30
    ret, frame = cap.read()
    #frames.append(frame)

    if i % 15 == 0 or True: 
        results = model(frame, imgsz=1280)[0]
        detections = sv.Detections.from_ultralytics(results)


    labels = [f"{model.names[detections.class_id[i]]} {detections.confidence[i]:0.2f}" for i in range(len(detections.class_id))]
    #print(detections.class_id)
    if person in detections.class_id and time.time() - lastBeep > 4:
        winsound.Beep(frequency, duration)
        lastBeep = time.time()

    frame = box_annotator.annotate(scene=frame, detections=detections, labels=labels)

    startTime = time.time()
    frame = cv2.resize(frame, (1920, 1080))
    #frame = cv2.cuda.resize(frame,(1920,1080),interpolation=cv2.INTER_LINEAR)
    #frame = torch.nn.functional.interpolate(frame,size=(1920,1080),mode="bilinear")
    #time2 = time.time()
    cv2.imshow('frame', frame)
    #print((time2 - startTime)*1000, "ms", "    ", (time.time() - time2)*1000, "ms")
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    
    frameTimes.append(time.time())
    while time.time() - frameTimes[0] >= 10:    
        frameTimes.pop(0)

    print("FPS =", len(frameTimes) / 10)
    i+=1

    

cap.release()
cv2.destroyAllWindows()

time.sleep(3)

'''
print("start resize")
for frame in frames:
    frame2 = cv2.resize(frame, (1920, 1080))
print("end resize")'''

#sv.process_video(source_path=VIDEO_PATH, target_path=f"result.mp4", callback=process_frame)
