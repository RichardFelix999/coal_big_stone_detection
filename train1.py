import warnings
warnings.filterwarnings('ignore')
from ultralytics import YOLO

if __name__ == '__main__':
    # Load a YOLOv8 model (choose yolov8n.pt, yolov8s.pt, etc. as needed)
    model = YOLO('yolov8n.pt')  # Or yolov8s.pt, yolov8m.pt, etc.

    # Train the model
    model.train(
        data='coal_stone.yaml',      # Path to your YAML config
        task='detect',               # Detection task
        imgsz=640,                   # Image size
        epochs=10,                  # Number of epochs
        batch=16,                    # Batch size
        device='cpu',                  # GPU device (set 'cpu' for CPU training)
        workers=8,                   # Number of data loader workers
        optimizer='SGD',             # Optimizer (SGD or Adam)
        project='runs/train',        # Project directory
        name='big_stone_exp',        # Experiment name
        cache=False,                 # Don't cache images in RAM
        close_mosaic=10,             # Disable mosaic augmentation after 10 epochs
        #patience=15,                # Early stopping patience (uncomment if needed)
        #resume='',                  # Path to resume checkpoint (uncomment if needed)
        #amp=False,                  # Disable automatic mixed precision (uncomment if needed)
    )
