import cv2
from ultralytics import YOLO

def detect_video(input_path, output_path, model_path, conf_threshold=0.25, device='cpu'):
    # Load the YOLOv8 model
    model = YOLO(model_path)

    # Open the input video
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        print(f"Error opening video file {input_path}")
        return

    # Get video properties for output video writer
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    # Define the codec and create VideoWriter object to save output video
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # or 'XVID', 'MJPG', etc.
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    while True:
        ret, frame = cap.read()
        if not ret:
            break  # End of video

        # Run detection on the frame
        results = model.predict(source=frame, conf=conf_threshold, device=device, save=False)

        # results is a list; get the first (and only) result
        annotated_frame = results[0].plot()  # Draw bounding boxes on the frame

        # Write the annotated frame to output video
        out.write(annotated_frame)

        # Optional: display frame in a window
        cv2.imshow('YOLOv8 Detection', annotated_frame)
        if cv2.waitKey(1) & 0xFF == 27:  # Press ESC to exit early
            break

    # Release resources
    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print(f"Output video saved to {output_path}")

if __name__ == "__main__":
    input_video = 'E:/essay/FSFL-GW-yolo/datasets/videos/input_video.mp4'  # Your input video path
    output_video = 'E:/essay/FSFL-GW-yolo/datasets/videos/output_marked.mp4' # Where to save marked video
    model_weights = 'runs/train/big_stone_exp5/weights/best.pt'               # Your trained model weights

    detect_video(input_video, output_video, model_weights, conf_threshold=0.25, device='cpu')
