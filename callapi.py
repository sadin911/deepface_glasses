# -*- coding: utf-8 -*-
"""
Created on Tue Mar 25 14:02:31 2025

@author: sadin
"""

import requests
import base64

# Read and encode an image to base64
with open("test_image.jpg", "rb") as image_file:
    base64_image = base64.b64encode(image_file.read()).decode('utf-8')

# Prepare request payload
payload = {
    "base64image": base64_image,
    "min_detection_threshold": 0.5,
    "glasses_detection_threshold": 0.1
}

# Send request
response = requests.post("http://localhost:5000/detect_glasses", json=payload)

# Print result
print(response.json())