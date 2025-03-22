import cv2

cap = cv2.VideoCapture(0, cv2.CAP_V4L2)

if not cap.isOpened():
    print("❌ Failed to open webcam")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("❌ Failed to capture frame")
        break
    cv2.imshow("Webcam Test", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
