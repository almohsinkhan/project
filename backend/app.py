from flask import Flask, Response
import cv2
from ultralytics import YOLO

app = Flask(__name__)

# Load YOLO model
model = YOLO("yolov8n.pt")

# Detect available camera (DroidCam usually at /dev/video10 or higher)
# Detect available camera (DroidCam usually at /dev/video10 or higher)
droidcam_index = None
for i in range(15):  # Check multiple indexes
    cap = cv2.VideoCapture(i)
    if cap.isOpened():
        print(f"✅ Camera found at index {i}")
        droidcam_index = i  # Assign the detected index
        cap.release()
        break  # Stop at the first working camera

# Ensure we found a camera
if droidcam_index is None:
    raise Exception("❌ No available camera found! Make sure DroidCam is running.")

# Open the detected camera
camera = cv2.VideoCapture(droidcam_index)


def generate_frames():
    while True:
        success, frame = camera.read()
        if not success:
            print("❌ Failed to capture frame. Stopping stream.")
            break  # Stop loop if frame capture fails

        # Perform YOLO detection
        results = model(frame)
        for result in results:
            frame = result.plot()  # Draw bounding boxes

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
