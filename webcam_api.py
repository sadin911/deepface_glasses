import cv2
import numpy as np
import requests
import base64
import json
from mediapipe.python.solutions import face_detection as mp_face_detection

# API endpoint (adjust IP if Docker is on a different machine)
API_URL = "http://localhost:8000/detect-glasses/"

# Initialize MediaPipe Face Detection
mp_face_detection = mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.5)

def detect_glasses(frame):
    try:
        # Convert frame to RGB for MediaPipe
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Detect faces
        results = mp_face_detection.process(rgb_frame)
        
        if results.detections:
            # Use the first detected face
            detection = results.detections[0]
            bbox = detection.location_data.relative_bounding_box
            
            # Get frame dimensions
            h, w, _ = frame.shape
            
            # Convert relative coordinates to absolute
            xmin = int(bbox.xmin * w)
            ymin = int(bbox.ymin * h)
            width = int(bbox.width * w)
            height = int(bbox.height * h)
            
            # Add more padding (50% of face size)
            padding_x = int(width * 0.7)
            padding_y = int(height * 0.7)
            xmin = max(0, xmin - padding_x)
            ymin = max(0, ymin - padding_y)
            xmax = min(w, xmin + width + 2 * padding_x)
            ymax = min(h, ymin + height + 2 * padding_y)
            
            # Crop face region with more padding
            face_crop = frame[ymin:ymax, xmin:xmax]
            if face_crop.size == 0:
                return False, 0.0, None
            
            # Convert cropped frame to base64
            _, buffer = cv2.imencode('.jpg', face_crop)
            base64_image = base64.b64encode(buffer).decode('utf-8')
            
            # Send to API
            payload = {"image": base64_image}
            response = requests.post(API_URL, json=payload, timeout=5)
            response.raise_for_status()  # Raise exception for bad status codes
            
            # Parse API response
            result = response.json()
            has_glasses = result["isGlasses"]
            confidence = result["score"]
            
            return has_glasses, confidence, (xmin, ymin, xmax, ymax)
        else:
            return False, 0.0, None  # No face detected
    except Exception as e:
        print(f"Error: {str(e)}")
        return False, 0.0, None

def run_webcam_detection():
    # Open webcam
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Error: Could not open webcam.")
    
    print("Press 'q' to quit.")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to capture frame.")
            break
        
        # Detect glasses via API
        has_glasses, confidence, bbox = detect_glasses(frame)
        
        # Display result
        if bbox is not None:
            xmin, ymin, xmax, ymax = bbox
            cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
            label = "Glasses" if has_glasses else "No Glasses"
            text = f"{label}: {confidence:.4f}"
            color = (0, 255, 0) if has_glasses else (0, 0, 255)
            cv2.putText(frame, text, (xmin, ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
        else:
            cv2.putText(frame, "No Face Detected", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)
        
        # Show frame
        cv2.imshow('Glasses Detection', frame)
        
        # Quit on 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Cleanup
    cap.release()
    cv2.destroyAllWindows()
    mp_face_detection.close()
    print("Webcam detection stopped.")

if __name__ == "__main__":
    try:
        run_webcam_detection()
    except Exception as e:
        print(f"Error: {str(e)}")