import os
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import cv2
import numpy as np
import mediapipe as mp

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Load trained model from checkpoint
CHECKPOINT_PATH = './glasses_detector_checkpoint.pth'
model = models.mobilenet_v2(weights=None)
model.classifier[1] = nn.Linear(model.last_channel, 2)  # 2 classes: glasses, no_glasses
if os.path.exists(CHECKPOINT_PATH):
    model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=device))
    print(f"Loaded model checkpoint from {CHECKPOINT_PATH}")
else:
    raise FileNotFoundError(f"Checkpoint not found at {CHECKPOINT_PATH}")
model = model.to(device)
model.eval()

# Preprocessing pipeline (matches training)
preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Initialize MediaPipe Face Detection
mp_face_detection = mp.solutions.face_detection
face_detection = mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.5)  # Full-range model

# Glasses detection function with face cropping
def detect_glasses(frame):
    try:
        # Convert BGR to RGB for MediaPipe
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Detect faces
        results = face_detection.process(rgb_frame)
        
        if results.detections:
            # Use the first detected face (assumes one primary face)
            detection = results.detections[0]
            bbox = detection.location_data.relative_bounding_box
            
            # Get frame dimensions
            h, w, _ = frame.shape
            
            # Convert relative coordinates to absolute
            xmin = int(bbox.xmin * w)
            ymin = int(bbox.ymin * h)
            width = int(bbox.width * w)
            height = int(bbox.height * h)
            
            # Add padding (e.g., 20% of face size) for better context
            padding_x = int(width * 0.2)
            padding_y = int(height * 0.2)
            xmin = max(0, xmin - padding_x)
            ymin = max(0, ymin - padding_y)
            xmax = min(w, xmin + width + 2 * padding_x)
            ymax = min(h, ymin + height + 2 * padding_y)
            
            # Crop face region
            face_crop = frame[ymin:ymax, xmin:xmax]
            if face_crop.size == 0:  # Check if crop is valid
                return False, 0.0, None
            
            # Convert cropped face to RGB PIL image
            rgb_face = cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(rgb_face)
            
            # Preprocess and run inference
            img_tensor = preprocess(pil_image).unsqueeze(0).to(device)
            with torch.no_grad():
                outputs = model(img_tensor)
                probabilities = torch.softmax(outputs, dim=1)
                confidence, predicted = torch.max(probabilities, 1)
                has_glasses = predicted.item() == 0  # Assuming glasses=0 from class_to_idx
            
            return has_glasses, confidence.item(), (xmin, ymin, xmax, ymax)
        else:
            return False, 0.0, None  # No face detected
    except Exception as e:
        print(f"Error processing frame: {str(e)}")
        return False, 0.0, None

# Webcam setup and real-time detection
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
        
        # Detect glasses with face cropping
        has_glasses, confidence, bbox = detect_glasses(frame)
        
        # Display result
        if bbox is not None:
            xmin, ymin, xmax, ymax = bbox
            # Draw bounding box around face
            cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
            
            # Prepare text
            label = "Glasses" if has_glasses else "No Glasses"
            text = f"{label}: {confidence:.4f}"
            color = (0, 255, 0) if has_glasses else (0, 0, 255)
            
            # Position text above the bounding box
            cv2.putText(frame, text, (xmin, ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2, cv2.LINE_AA)
        else:
            # No face detected
            cv2.putText(frame, "No Face Detected", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2, cv2.LINE_AA)
        
        # Show frame
        cv2.imshow('Glasses Detection', frame)
        
        # Quit on 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Cleanup
    cap.release()
    cv2.destroyAllWindows()
    face_detection.close()
    print("Webcam detection stopped.")

if __name__ == "__main__":
    try:
        run_webcam_detection()
    except Exception as e:
        print(f"Error: {str(e)}")