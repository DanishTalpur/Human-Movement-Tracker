from ultralytics import YOLO
import cv2
import numpy as np

# Load model
model = YOLO(r"Training\runs\detect\train\weights\best.pt")

# Open video
cap = cv2.VideoCapture(r"videos\background-video-people-walking_1080p.mp4")

# Get input video properties
fps = cap.get(cv2.CAP_PROP_FPS)
width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Define VideoWriter
fourcc = cv2.VideoWriter_fourcc(*"XVID")
out = cv2.VideoWriter("videos\output_tracking_people.avi", fourcc, fps, (width, height))

# Store last positions and smoothed directions
track_history = {}
direction_history = {}
last_seen_arrow = {}
arrow_persistence = 10   # frames to keep arrow if movement drops
min_speed = 2            # threshold for movement detection

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Run tracking
    results = model.track(frame, persist=True)
    r = results[0]

    boxes = r.boxes.xyxy.cpu().numpy()
    ids = r.boxes.id.int().cpu().tolist() if r.boxes.id is not None else []

    # Create overlay for transparent arrows
    overlay = frame.copy()

    for box, track_id in zip(boxes, ids):
        x1, y1, x2, y2 = map(int, box)
        box_color = (255, 88, 230)   # purple box
        arrow_color = (140, 50, 255)    # pink arrow

        # Draw bounding box
        cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 3)
        label = f"Person {track_id}"
        cv2.putText(frame, label, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, box_color, 2)

        # Current center
        cx = int((x1 + x2) / 2)
        cy = int((y1 + y2) / 2)

        # Store previous point
        if track_id not in track_history:
            track_history[track_id] = (cx, cy)
            direction_history[track_id] = np.array([0, 0], dtype=float)
            last_seen_arrow[track_id] = 0
            continue

        prev_cx, prev_cy = track_history[track_id]
        track_history[track_id] = (cx, cy)

        # Raw displacement vector
        dx = cx - prev_cx
        dy = cy - prev_cy

        # Update smoothed direction with EMA
        prev_dir = direction_history[track_id]
        new_dir = np.array([dx, dy], dtype=float)
        smoothed_dir = 0.8 * prev_dir + 0.2 * new_dir
        direction_history[track_id] = smoothed_dir

        # Speed (vector magnitude)
        speed = np.linalg.norm(smoothed_dir)

        # If speed too low, decrement persistence counter
        if speed < min_speed:
            last_seen_arrow[track_id] -= 1
        else:
            last_seen_arrow[track_id] = arrow_persistence  # reset counter

        # If still within persistence window, draw arrow
        if last_seen_arrow[track_id] > 0:
            unit_dir = smoothed_dir / (speed + 1e-6)
            arrow_len = 60
            end_point = (int(cx + unit_dir[0] * arrow_len),
                         int(cy + unit_dir[1] * arrow_len))

            cv2.arrowedLine(
                overlay,
                (cx, cy), end_point,
                arrow_color,
                thickness=8,    # bold
                tipLength=0.5   # big head
            )

    # Blend overlay with transparency
    alpha = 0.5
    frame = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)

    out.write(frame)

cap.release()
out.release()
print("Saved output video(persistent arrows for movers)")
