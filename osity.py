"""
OSITY - Object Detection, Segmentation, and Classification Model
This script implements YOLO11m-seg model for detecting, classifying, counting,
and estimating proportions of particles in images and videos.
"""

import cv2
import os
from ultralytics import YOLO

def setup_model(model_path):
    """Initialize the YOLO model"""
    return YOLO(model_path)

def run_inference_basic(model, source, save=True, imgsz=640, conf=0.7):
    """Run basic inference with object counting and percentage calculation"""

    # Create output directory if it doesn't exist
    output_dir = "runs/segment"
    os.makedirs(output_dir, exist_ok=True)

    # Run prediction (don't let YOLO create a separate project folder; we'll save custom result)
    results = model.predict(
        source=source,
        save=False,
        imgsz=imgsz,
        conf=conf
    )[0]

    # Get detections
    detections = results.boxes

    # Filter detections with high confidence
    confident_detections = [box for box in detections if float(box.conf) >= conf]

    # Count objects per class
    class_counts = {}
    total_count = 0
    for box in confident_detections:
        class_name = results.names[int(box.cls)]
        class_counts[class_name] = class_counts.get(class_name, 0) + 1
        total_count += 1

    # Get the result image with YOLO's default visualization
    result_image = results.plot()

    # Create summary lines that include percentages
    summary_lines = []
    summary_lines.append(f"Total objects: {total_count}")
    for class_name, count in class_counts.items():
        pct = (count / total_count * 100) if total_count > 0 else 0.0
        summary_lines.append(f"{class_name}: {count} ({pct:.1f}%)")

    # Add semi-transparent summary bar at the top
    height = len(summary_lines) * 40 + 20
    overlay = result_image.copy()
    cv2.rectangle(overlay, (0, 0), (result_image.shape[1], height), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.3, result_image, 0.7, 0, result_image)

    # Add text to the summary bar
    y_position = 30
    for text in summary_lines:
        cv2.putText(result_image, text, (20, y_position), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        y_position += 40

    # Save result with counts
    output_path = os.path.join(output_dir, "output_with_counts.jpg")
    cv2.imwrite(output_path, result_image)

    # Briefly display result and then close windows
    cv2.imshow("Detection Results", result_image)
    cv2.waitKey(2000)
    cv2.destroyAllWindows()

    # Print counts to console (with percentages)
    print("\nObject Counts Summary:")
    print(f"Total objects detected: {total_count}")
    for class_name, count in class_counts.items():
        pct = (count / total_count * 100) if total_count > 0 else 0.0
        print(f"{class_name}: {count} ({pct:.1f}%)")

def main():
    # PATHS
    MODEL_PATH = r'C:/Path to the model'
    SOURCE_PATH = r'C:/Path to the Image'
    
    # Initialize model
    model = setup_model(MODEL_PATH)
    
    # Run inference with basic counting
    print("Running inference and counting...")
    run_inference_basic(model, SOURCE_PATH)

if __name__ == "__main__":
    main()
