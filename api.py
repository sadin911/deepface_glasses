# -*- coding: utf-8 -*-
"""
Created on Tue Mar 25 14:00:45 2025

@author: sadin
"""

from flask import Flask, request, jsonify
import cv2
import mediapipe as mp
import numpy as np
import base64
from io import BytesIO
from PIL import Image

app = Flask(__name__)

# Initialize MediaPipe Face Detection
mp_face_detection = mp.solutions.face_detection

def process_image(base64_string):
    # Decode base64 image
    img_data = base64.b64decode(base64_string)
    img = Image.open(BytesIO(img_data))
    # Convert PIL image to numpy array (BGR format for OpenCV)
    frame = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    return frame

def detect_glasses(frame, detection, glasses_threshold):
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
    right_eye = keypoints[1] # Left eye (right in image)
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
        return 0.0, glasses_threshold, False
    
    # Convert to grayscale, blur to reduce noise, and apply edge detection
    gray = cv2.cvtColor(eye_region, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)
    
    # Count edge pixels and calculate density
    edge_count = np.sum(edges > 0)
    area = (eye_x2 - eye_x1) * (eye_y2 - eye_y1)
    edge_density = edge_count / area if area > 0 else 0
    
    # Determine if glasses are present
    has_glasses = edge_density > glasses_threshold
    
    return edge_density, glasses_threshold, has_glasses

@app.route('/detect_glasses', methods=['POST'])
def detect_glasses_endpoint():
    try:
        # Get JSON data from request
        data = request.get_json()
        
        # Extract parameters
        base64_image = data.get('base64image')
        min_detection_conf = float(data.get('min_detection_threshold', 0.5))
        glasses_threshold = float(data.get('glasses_detection_threshold', 0.1))
        
        if not base64_image:
            return jsonify({'error': 'No image provided'}), 400
            
        # Process the image
        frame = process_image(base64_image)
        
        # Initialize face detection with provided threshold
        face_detection = mp_face_detection.FaceDetection(min_detection_confidence=min_detection_conf)
        
        # Convert to RGB for MediaPipe
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Process frame for face detection
        results = face_detection.process(rgb_frame)
        
        response = {
            'results': []
        }
        
        if results.detections:
            for i, detection in enumerate(results.detections):
                edge_density, threshold, has_glasses = detect_glasses(frame, detection, glasses_threshold)
                
                result = {
                    'face_id': i,
                    'edge_density': float(edge_density),
                    'threshold': float(threshold),
                    'has_glasses': has_glasses
                }
                response['results'].append(result)
        else:
            response['results'].append({
                'face_id': 0,
                'edge_density': 0.0,
                'threshold': float(glasses_threshold),
                'has_glasses': False,
                'message': 'No faces detected'
            })
            
        # Clean up
        face_detection.close()
        
        return jsonify(response), 200
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)