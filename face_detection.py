import cv2
import mediapipe as mp
import numpy as np

# Initialize MediaPipe Face Detection
mp_face_detection = mp.solutions.face_detection
face_detection = mp_face_detection.FaceDetection(min_detection_confidence=1)
mp_draw = mp.solutions.drawing_utils

# Open webcam
cap = cv2.VideoCapture(0)

def detect_glasses(frame, detection):
    # Get image dimensions
    h, w, _ = frame.shape
    
    # Extract bounding box and keypoints
    bbox = detection.location_data.relative_bounding_box
    keypoints = detection.location_data.relative_keypoints
    
    # Convert normalized coordinates to pixel values
    x_min = int(bbox.xmin * w)
    y_min = int(bbox.ymin * h)
    box_w = int(bbox.width * w)
    box_h = int(bbox.height * h)
    
    # Focus on eye region (approximate from keypoints)
    left_eye = keypoints[0]  # Right eye (left in image)
    right_eye = keypoints[1]  # Left eye (right in image)
    eye_x1 = int(left_eye.x * w) - 20
    eye_y1 = int(left_eye.y * h) - 20
    eye_x2 = int(right_eye.x * w) + 20
    eye_y2 = int(right_eye.y * h) + 20
    
    # Ensure coordinates are within bounds
    eye_x1 = max(0, eye_x1)
    eye_y1 = max(0, eye_y1)
    eye_x2 = min(w, eye_x2)
    eye_y2 = min(h, eye_y2)
    
    # Extract eye region
    eye_region = frame[eye_y1:eye_y2, eye_x1:eye_x2]
    if eye_region.size == 0:
        return False
    
    # Convert to grayscale, blur to reduce noise, and apply edge detection
    gray = cv2.cvtColor(eye_region, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)  # Reduce noise
    edges = cv2.Canny(blurred, 50, 150)
    
    # Count edge pixels and calculate density
    edge_count = np.sum(edges > 0)
    area = (eye_x2 - eye_x1) * (eye_y2 - eye_y1)
    edge_density = edge_count / area if area > 0 else 0
    
    # Higher threshold to be stricter about glasses detection
    glasses_threshold = 0.1  # Increased from 0.05
    has_glasses = edge_density > glasses_threshold
    
    # Debugging: Show edge density and eye region
    print(f"Edge Density: {edge_density:.4f}, Threshold: {glasses_threshold}")
    cv2.imshow('Eye Region', eye_region)
    cv2.imshow('Edges', edges)
    
    # Draw rectangle around eye region
    cv2.rectangle(frame, (eye_x1, eye_y1), (eye_x2, eye_y2), (0, 255, 255), 2)
    
    return has_glasses

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read frame.")
        break

    # Convert BGR to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process frame for face detection
    results = face_detection.process(rgb_frame)

    # Draw face detections and check for glasses
    if results.detections:
        for detection in results.detections:
            mp_draw.draw_detection(frame, detection)
            has_glasses = detect_glasses(frame, detection)
            
            # Label the face
            bbox = detection.location_data.relative_bounding_box
            x_min = int(bbox.xmin * frame.shape[1])
            y_min = int(bbox.ymin * frame.shape[0])
            label = "Glasses" if has_glasses else "No Glasses"
            cv2.putText(frame, label, (x_min, y_min - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    # Show the frame
    cv2.imshow('MediaPipe Face Detection with Glasses', frame)

    # Exit on 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()
face_detection.close()