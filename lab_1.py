from ultralytics import YOLO

# Load the model
model = YOLO("yolov8n.pt")

# Set source to 0 for Webcam, or the filename for images/videos
source = 0 

# Run the detection
# We add 'show=True' so the window pops up automatically
results = model.predict(source=source, show=True, save=True, conf=0.35)

print("Webcam test active. Press 'q' on your keyboard to stop.")