import os
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import cv2
import numpy as np
import mediapipe as mp
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import base64
import io
import uvicorn

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
face_detection = mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.5)

# Define request model for base64 input
class ImageRequest(BaseModel):
    image: str  # Base64-encoded image string

# Glasses detection function (with face cropping, matching webcam)
def detect_glasses(base64_image: str):
    try:
        # Decode base64 string to image
        image_data = base64.b64decode(base64_image)
        nparr = np.frombuffer(image_data, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        # Convert BGR to RGB for MediaPipe
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Detect faces
        results = face_detection.process(rgb_frame)
        
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
            
            # Add padding (20% of face size, matching webcam)
            padding_x = int(width * 0.2)
            padding_y = int(height * 0.2)
            xmin = max(0, xmin - padding_x)
            ymin = max(0, ymin - padding_y)
            xmax = min(w, xmin + width + 2 * padding_x)
            ymax = min(h, ymin + height + 2 * padding_y)
            
            # Crop face region
            face_crop = frame[ymin:ymax, xmin:xmax]
            if face_crop.size == 0:
                return {"isGlasses": False, "score": 0.0}
            
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
            
            return {"isGlasses": has_glasses, "score": float(confidence.item())}
        else:
            return {"isGlasses": False, "score": 0.0}  # No face detected
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing image: {str(e)}")

# Initialize FastAPI app
app = FastAPI(title="Glasses Detection API")

# API endpoint to detect glasses from base64 image
@app.post("/detect-glasses/")
async def detect_glasses_endpoint(request: ImageRequest):
    result = detect_glasses(request.image)
    return JSONResponse(content=result)

# Health check endpoint
@app.get("/health")
async def health_check():
    return {"status": "healthy", "device": str(device)}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)