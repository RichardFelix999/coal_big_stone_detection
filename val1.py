import warnings
warnings.filterwarnings('ignore')
from ultralytics import YOLO

if __name__ == '__main__':
    # Load your trained model (replace with actual path)
    model = YOLO('runs/train/big_stone_exp5/weights/best.pt')  # NOT 'best (1).pt'
    
    # Validate
    model.val(
        data='coal_stone.yaml',          # Your dataset config
        split='val',                     # Use 'test' for final evaluation
        imgsz=640,                       # Match training size
        batch=16,
        conf=0.5,                        # Confidence threshold
        iou=0.6,                         # IoU threshold
        device='cpu',                    # Match training device
        plots=True,                      # Generate confusion matrix/PR curves
        save_json=True,                  # For COCO-style metrics
        project='runs/val',              # Save location
        name='big_stone_val',            # Unique experiment name
        exist_ok=False                   # Prevent overwriting
    )
