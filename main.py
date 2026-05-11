import cv2
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort

# Load YOLO model
model = YOLO("yolov8n.pt")

# Initialize tracker
tracker = DeepSort(max_age=30)

# Open webcam
cap = cv2.VideoCapture(0)

while True:

    ret, frame = cap.read()

    if not ret:
        break

    # Run YOLO detection
    results = model(frame)

    detections = []

    for result in results:

        boxes = result.boxes

        for box in boxes:

            x1, y1, x2, y2 = box.xyxy[0]

            x1 = int(x1)
            y1 = int(y1)
            x2 = int(x2)
            y2 = int(y2)

            confidence = float(box.conf[0])

            class_id = int(box.cls[0])

            class_name = model.names[class_id]

            detections.append(
                (
                    [x1, y1, x2 - x1, y2 - y1],
                    confidence,
                    class_name
                )
            )

    # Tracking
    tracks = tracker.update_tracks(
        detections,
        frame=frame
    )

    # Draw boxes
    for track in tracks:

        if not track.is_confirmed():
            continue

        track_id = track.track_id

        ltrb = track.to_ltrb()

        x1, y1, x2, y2 = map(int, ltrb)

        cv2.rectangle(
            frame,
            (x1, y1),
            (x2, y2),
            (0, 255, 0),
            2
        )

        cv2.putText(
            frame,
            f"ID: {track_id}",
            (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2
        )

    cv2.imshow(
        "Object Detection and Tracking",
        frame
    )

    # Press q to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()