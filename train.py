import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms, datasets
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")
if device.type == 'cuda':
    print(f"GPU Name: {torch.cuda.get_device_name(0)}")

# Hyperparameters
NUM_EPOCHS = 50
BATCH_SIZE = 32
LEARNING_RATE = 0.001
DATA_DIR = './data'
CHECKPOINT_PATH = './glasses_detector_checkpoint.pth'
MODEL_PATH = './glasses_detector_full.pth'
LOG_DIR = './runs'
LOG_FILE = os.path.join(LOG_DIR, 'glasses_detection_events')

# Data preprocessing
data_transforms = {
    'train': transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]),
    'val': transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]),
}

def prepare_datasets(data_dir):
    full_dataset = datasets.ImageFolder(root=data_dir, transform=data_transforms['train'])
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(full_dataset, [train_size, val_size])
    
    val_dataset.dataset.transform = data_transforms['val']
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
    
    print(f"Training samples: {len(train_dataset)}, Validation samples: {len(val_dataset)}")
    return train_loader, val_loader, full_dataset.class_to_idx

# Model setup
model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)
model.classifier[1] = nn.Linear(model.last_channel, 2)
model = model.to(device)

# Load existing checkpoint if available
if os.path.exists(CHECKPOINT_PATH):
    model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=device))
    print(f"Loaded checkpoint from {CHECKPOINT_PATH}")

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

# TensorBoard writer with single log file and real-time flushing
os.makedirs(LOG_DIR, exist_ok=True)
writer = SummaryWriter(log_dir=LOG_FILE)
print(f"Logging to TensorBoard file: {LOG_FILE}")

def train_model(train_loader, val_loader):
    global_step = 0
    
    for epoch in range(NUM_EPOCHS):
        model.train()
        epoch_loss = 0.0
        epoch_correct = 0
        epoch_total = 0
        
        # Training phase
        for i, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)
            if epoch == 0 and i == 0:
                print(f"First batch - Images on {images.device}, Labels on {labels.device}")
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs.data, 1)
            epoch_total += labels.size(0)
            epoch_correct += (predicted == labels).sum().item()
            
            # Log step-level metrics every 10 steps (adjustable)
            global_step += 1
            if (i + 1) % 10 == 0:
                step_loss = loss.item()
                step_accuracy = 100 * (predicted == labels).sum().item() / labels.size(0)
                print(f'Epoch [{epoch+1}/{NUM_EPOCHS}], Step [{i+1}/{len(train_loader)}], '
                      f'Loss: {step_loss:.4f}, Accuracy: {step_accuracy:.2f}%')
                writer.add_scalar('training_loss_step', step_loss, global_step)
                writer.add_scalar('training_accuracy_step', step_accuracy, global_step)
                writer.flush()  # Flush to disk for real-time updates
        
        # Log epoch-level training metrics
        avg_train_loss = epoch_loss / epoch_total
        train_accuracy = 100 * epoch_correct / epoch_total
        print(f'Epoch [{epoch+1}/{NUM_EPOCHS}], Train Loss: {avg_train_loss:.4f}, '
              f'Train Accuracy: {train_accuracy:.2f}%')
        writer.add_scalar('training_loss_epoch', avg_train_loss, epoch)
        writer.add_scalar('training_accuracy_epoch', train_accuracy, epoch)
        writer.flush()  # Flush epoch metrics
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item() * images.size(0)
                _, predicted = torch.max(outputs.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
        
        avg_val_loss = val_loss / val_total
        val_accuracy = 100 * val_correct / val_total
        print(f'Validation - Epoch [{epoch+1}/{NUM_EPOCHS}], Loss: {avg_val_loss:.4f}, '
              f'Accuracy: {val_accuracy:.2f}%')
        writer.add_scalar('validation_loss', avg_val_loss, epoch)
        writer.add_scalar('validation_accuracy', val_accuracy, epoch)
        writer.flush()  # Flush validation metrics
        
        # Save checkpoint after every epoch
        torch.save(model.state_dict(), CHECKPOINT_PATH)
        print(f"Checkpoint saved at epoch {epoch+1} to '{CHECKPOINT_PATH}'")
    
    # Save full model at the end
    torch.save(model, MODEL_PATH)
    print(f"Full model saved as '{MODEL_PATH}'")

if __name__ == "__main__":
    train_loader, val_loader, class_to_idx = prepare_datasets(DATA_DIR)
    print(f"Classes: {class_to_idx}")
    
    train_model(train_loader, val_loader)
    writer.close()