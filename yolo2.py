import torch
from ultralytics import YOLO
from PIL import Image
import cv2
import supervision as sv
import numpy as np
import time


VIDEO_PATH = r"D:\Johnathan\Videos\2024-03-17 22-54-56.mp4";

# Check for CUDA device and set it
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Using device: {device}')

# Load model
model = YOLO('yolov8n.pt').to(device)

video_info = sv.VideoInfo.from_video_path(VIDEO_PATH)
def process_frame(frame: np.ndarray, _) -> np.ndarray:
    results = model(frame, imgsz=1280)[0]
    
    detections = sv.Detections.from_ultralytics(results)

    box_annotator = sv.BoxAnnotator(thickness=2, text_thickness=1, text_scale=0.5)

    print(len(detections))

    #labels = [f"{model.names[class_id]} {confidence:0.2f}" for _, _, confidence, class_id, _ in detections]

    labels = [f"{model.names[detections.class_id[i]]} {detections.confidence[i]:0.2f}" for i in range(len(detections.class_id))]


    frame = box_annotator.annotate(scene=frame, detections=detections, labels=labels)

    return frame

startTime = time.time()
sv.process_video(source_path=VIDEO_PATH, target_path=f"result.mp4", callback=process_frame)
print("Ran for", time.time() - startTime, "seconds")

#sv.process_video(source_path=VIDEO_PATH, target_path=f"result.mp4", callback=process_frame)
