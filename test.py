import warnings
warnings.filterwarnings('ignore')
from ultralytics import YOLO
import glob
import os
import cv2

if __name__ == '__main__':
    # Path to your trained model weights
    MODEL_PATH = 'runs/train/big_stone_exp5/weights/best.pt'
    # Path to your test images folder
    TEST_IMAGES_DIR = 'E:/essay/FSFL-GW-yolo/datasets/images/test'
    # Output folder for marked images
    OUTPUT_DIR = 'marked_results'
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    # File types to test
    IMAGE_EXTENSIONS = ['*.jpg', '*.jpeg', '*.png', '*.bmp']

    # Collect all image paths in the test folder
    image_paths = []
    for ext in IMAGE_EXTENSIONS:
        image_paths.extend(glob.glob(os.path.join(TEST_IMAGES_DIR, ext)))

    if not image_paths:
        print(f"No images found in {TEST_IMAGES_DIR}")
        exit(1)

    # Load the trained YOLOv8 model
    model = YOLO(MODEL_PATH)

    for img_path in image_paths:
        # Run prediction
        results = model.predict(
            source=img_path,
            imgsz=640,
            conf=0.25,
            device='cpu',
            save=False  # We will save ourselves
        )

        # Visualize and save the result
        marked_img = results[0].plot()  # Get the image with boxes drawn
        # Save to output folder with the same filename
        filename = os.path.basename(img_path)
        save_path = os.path.join(OUTPUT_DIR, filename)
        cv2.imwrite(save_path, marked_img)
        print(f"Processed and saved: {save_path}")

    print("\nAll images processed and marked images saved in the 'marked_results' folder.")
