import os
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import shutil
from pathlib import Path

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load model from checkpoint
CHECKPOINT_PATH = './glasses_detector_checkpoint.pth'
model = models.mobilenet_v2(weights=None)
model.classifier[1] = nn.Linear(model.last_channel, 2)
model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=device))
model = model.to(device)
model.eval()

# Preprocessing
preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Directories
INPUT_DIR = r"D:\UTK\part1"
GLASSES_DIR = "./images/glasses"
NO_GLASSES_DIR = "./images/noglasses"

Path(GLASSES_DIR).mkdir(parents=True, exist_ok=True)
Path(NO_GLASSES_DIR).mkdir(parents=True, exist_ok=True)

def detect_glasses(image_path):
    try:
        img = Image.open(image_path).convert('RGB')
        img_tensor = preprocess(img).unsqueeze(0).to(device)
        
        with torch.no_grad():
            outputs = model(img_tensor)
            probabilities = torch.softmax(outputs, dim=1)
            confidence, predicted = torch.max(probabilities, 1)
            has_glasses = predicted.item() == 0  # Adjust based on class_to_idx (glasses is likely 0)
        
        return confidence.item(), has_glasses
    except Exception as e:
        print(f"Error processing {image_path}: {str(e)}")
        return 0.0, False

def categorize_images():
    image_extensions = ('.jpg', '.jpeg', '.png')
    processed_count = 0
    
    for filename in os.listdir(INPUT_DIR):
        if filename.lower().endswith(image_extensions):
            input_path = os.path.join(INPUT_DIR, filename)
            confidence, has_glasses = detect_glasses(input_path)
            
            confidence_prefix = f"{confidence:.4f}_"
            dest_dir = GLASSES_DIR if has_glasses else NO_GLASSES_DIR
            new_filename = confidence_prefix + filename
            dest_path = os.path.join(dest_dir, new_filename)
            
            try:
                shutil.copy2(input_path, dest_path)
                print(f"Categorized {filename} -> {dest_dir}/{new_filename}")
                processed_count += 1
            except Exception as e:
                print(f"Error copying {filename}: {str(e)}")
    
    print(f"\nProcessed {processed_count} images successfully.")

if __name__ == "__main__":
    categorize_images()