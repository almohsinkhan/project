from flask import Flask, Response
import cv2
from ultralytics import YOLO
import pyttsx3  # For speech output

app = Flask(__name__)

# Load YOLO model
model = YOLO("yolov8n.pt")

# Initialize text-to-speech engine
engine = pyttsx3.init()
engine.setProperty("rate", 120)  # Adjust speech speed
engine.setProperty("volume", 2.0)

# Detect available camera
droidcam_index = None
for i in range(15):  # Check multiple indexes
    cap = cv2.VideoCapture(i)
    if cap.isOpened():
        print(f"✅ Camera found at index {i}")
        droidcam_index = i
        cap.release()
        break  # Use first available camera

if droidcam_index is None:
    raise Exception("❌ No available camera found!")

# Open the camera
camera = cv2.VideoCapture(droidcam_index)

def speak(text):
    """Speak out the detected objects."""
    engine.say(text)
    engine.runAndWait()

def generate_frames():
    detected_objects = set()  # To avoid repeating the same object multiple times

    while True:
        success, frame = camera.read()
        if not success:
            print("❌ Failed to capture frame. Stopping stream.")
            break  # Stop loop if frame capture fails

        # Perform YOLO detection
        results = model(frame)

        for result in results:
            frame = result.plot()  # Draw bounding boxes
            
            # Get detected objects
            objects = {model.names[int(box.cls)] for box in result.boxes}
            
            # Announce new objects
            new_objects = objects - detected_objects
            if new_objects:
                speak(f"{', '.join(new_objects)}")
                detected_objects.update(new_objects)

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
