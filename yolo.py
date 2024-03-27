import cv2
import torch
from ultralytics import YOLO
from PIL import Image
#import supervision as sv
import numpy as np


# Check for CUDA device and set it
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Using device: {device}')

# Load model
model = YOLO('yolov8n.pt').to(device)

# Load a pretrained YOLO model (recommended for training)

#torch.cuda.set_device(0)

# Open the video file
video_path = r"D:\Johnathan\Videos\2024-03-17 22-54-56.mp4"
cap = cv2.VideoCapture(video_path)

i = 0
frames = []
# Loop through the video frames
while cap.isOpened():
    # Read a frame from the video
    success, frame = cap.read()

    if success:
        frames.append(frame)
    else:
        break


for frame in frames:
    # Run YOLOv8 inference on the frame
    results = model(frame)

    # Visualize the results on the frame
    annotated_frame = results[0].plot()

    # Display the annotated frame
    cv2.imshow("YOLOv8 Inference", annotated_frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Release the video capture object and close the display window
cap.release()
cv2.destroyAllWindows()

'''
im1 = Image.open("C:\\Users\\johna\\Downloads\\zz.jpg")

# Perform object detection on an image using the model
results = model(im1)

# Visualize the results
for i, r in enumerate(results):
    # Plot results image
    im_bgr = r.plot()  # BGR-order numpy array
    im_rgb = Image.fromarray(im_bgr[..., ::-1])  # RGB-order PIL image

    # Show results to screen (in supported environments)
    r.show()

    # Save results to disk
    fname=f'results{i}.jpg'
    r.save(filename=fname)
    print("saved as", fname) 

# Export the model to ONNX format
#success = model.export(format='onnx')
'''