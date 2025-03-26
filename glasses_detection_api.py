import cv2
import numpy as np
import requests
import base64
import json

# API endpoint (adjust IP if Docker is on a different machine)
API_URL = "http://localhost:8000/detect-glasses/"

def detect_glasses(frame):
    try:
        # Convert frame to base64 (full image)
        _, buffer = cv2.imencode('.jpg', frame)
        base64_image = base64.b64encode(buffer).decode('utf-8')
        
        # Send to API
        payload = {"image": base64_image}
        response = requests.post(API_URL, json=payload, timeout=5)
        response.raise_for_status()  # Raise exception for bad status codes
        
        # Parse API response
        result = response.json()
        has_glasses = result["isGlasses"]
        confidence = result["score"]
        
        # Since API crops, we don’t know the exact face bbox here
        # Return None for bbox and handle display accordingly
        return has_glasses, confidence, None
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
        
        # Detect glasses via API (full frame)
        has_glasses, confidence, _ = detect_glasses(frame)
        
        # Display result (no bounding box since API handles cropping)
        label = "Glasses" if has_glasses else "No Glasses"
        text = f"{label}: {confidence:.4f}"
        color = (0, 255, 0) if has_glasses else (0, 0, 255)
        cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
        
        # Show frame
        cv2.imshow('Glasses Detection', frame)
        
        # Quit on 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Cleanup
    cap.release()
    cv2.destroyAllWindows()
    print("Webcam detection stopped.")

if __name__ == "__main__":
    try:
        run_webcam_detection()
    except Exception as e:
        print(f"Error: {str(e)}")