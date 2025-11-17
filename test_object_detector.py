import cv2
from ultralytics.utils import LOGGER

# Disable all YOLO logs
LOGGER.setLevel(50)

from object_detector import ObjectDetector

# Load YOLO model
detector = ObjectDetector("yolov8n.pt")

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Detect objects
    detections = detector.detect(frame)

    # Draw detections
    for det in detections:
        x1, y1, x2, y2 = map(int, det["bbox"])
        label = det["label"]
        conf = det["conf"]

        # Draw bounding box
        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
        cv2.putText(frame, f"{label} {conf:.2f}",
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (255, 0, 0), 2)

    cv2.imshow("YOLO Object Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
