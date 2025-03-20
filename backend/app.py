from flask import Flask, Response
import cv2
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort
import pyttsx3
import time
import numpy as np

app = Flask(__name__)

# Load YOLO model
model = YOLO("yolov8l.pt")  # Using larger model for better accuracy

# Initialize DeepSORT tracker
tracker = DeepSort()

# Detect available camera (DroidCam usually at /dev/video10 or higher)
droidcam_index = None
for i in range(15):  # Check multiple indexes
    cap = cv2.VideoCapture(i)
    if cap.isOpened():
        print(f"✅ Camera found at index {i}")
        droidcam_index = i
        cap.release()
        break

if droidcam_index is None:
    raise Exception("❌ No available camera found! Make sure DroidCam is running.")

# Open the camera
camera = cv2.VideoCapture(droidcam_index)

# Initialize text-to-speech engine
engine = pyttsx3.init()
engine.setProperty("rate", 120)  # Slower speech for clarity

# Global tracking variables
last_speech_time = 0  # ✅ Fix: Initialize speech timing tracker
SPEECH_DELAY = 5  # Speak again only if 5 seconds have passed for the same object
last_detected_objects = {}  # ✅ Fix: Track last detection times for objects

def get_position(x1, x2, y1, y2, frame_width, frame_height):
    """Determine if an object is on the left, center, or right, and up, middle, or down."""
    
    # Horizontal position
    center_x = (x1 + x2) // 2
    if center_x < frame_width // 3:
        horizontal = "left"
    elif center_x > (2 * frame_width) // 3:
        horizontal = "right"
    else:
        horizontal = "center"
    
    # Vertical position
    center_y = (y1 + y2) // 2
    if center_y < frame_height // 3:
        vertical = "up"
    elif center_y > (2 * frame_height) // 3:
        vertical = "down"
    else:
        vertical = "middle"
    
    return horizontal, vertical

def generate_frames():
    global last_speech_time, last_detected_objects

    while True:
        success, frame = camera.read()
        if not success:
            print("❌ Failed to capture frame. Stopping stream.")
            break

        frame_height, frame_width, _ = frame.shape

        # Perform YOLO detection
        results = model(frame)

        detections = []
        object_count = {}  # Count objects by type
        object_positions = {}  # Store spatial awareness
        detected_labels = set()  # Track currently detected objects

        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()  # ✅ Fix: Ensure tensor is on CPU
                conf = box.conf[0].cpu().item()
                cls = int(box.cls[0].cpu().item())
                label = model.names[cls]

                if conf > 0.4:  # Confidence threshold
                    detections.append(([x1, y1, x2, y2], conf, label))
                    detected_labels.add(label)

                    # Count object type
                    object_count[label] = object_count.get(label, 0) + 1

                    # Determine position
                    position_h, position_v = get_position(x1, x2, y1, y2, frame_width, frame_height)
                    object_positions[label] = f"{position_v} {position_h}"  # Example: "middle right"

        # Update tracker
        tracks = tracker.update_tracks(detections, frame=frame)

        # **Context-Aware Speech Logic**
        current_time = time.time()
        speech_output = []

        for label, count in object_count.items():
            position = object_positions.get(label, "somewhere")

            # ✅ Fix: Check individual object timing instead of global delay
            if label not in last_detected_objects or (current_time - last_detected_objects[label] > SPEECH_DELAY):
                if count > 1:
                    speech_output.append(f"{count} {label}s detected, mostly on the {position}.")
                else:
                    speech_output.append(f"A {label} is on the {position}.")

                last_detected_objects[label] = current_time  # ✅ Update detection time for this object

        if speech_output:
            speech_text = " ".join(speech_output)
            print(f"🔊 Speaking: {speech_text}")
            engine.say(speech_text)
            engine.runAndWait()

        # Draw bounding boxes
        for track in tracks:
            if not track.is_confirmed():
                continue
            x1, y1, x2, y2 = map(int, track.to_ltrb())
            label = track.get_det_class()
            track_id = track.track_id

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"{label} ID:{track_id}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Encode frame as JPEG
        _, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/')
def index():
    return """<html>
                <head>
                    <title>Live Video Stream</title>
                </head>
                <body>
                    <h1>Live Object Detection Stream</h1>
                    <img src="/video_feed" width="800">
                </body>
              </html>"""

if __name__ == '__main__':
    app.run(debug=True)