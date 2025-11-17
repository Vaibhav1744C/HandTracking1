from ultralytics import YOLO

class ObjectDetector:
    def __init__(self, model_path="yolov8n.pt"):
        print("[INFO] Loading YOLO model...")
        self.model = YOLO(model_path)

    def detect(self, frame):
        """
        Returns: list of detections
        Each detection is a dict:
        {
            "bbox": [x1, y1, x2, y2],
            "label": class_name,
            "conf": confidence
        }
        """
        results = self.model(frame)
        
        detections = []

        for r in results:
            for box in r.boxes:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])
                label = self.model.names[cls_id]

                detections.append({
                    "bbox": [x1, y1, x2, y2],
                    "label": label,
                    "conf": conf
                })

        return detections
