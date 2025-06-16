from ultralytics import YOLO
import os
from pathlib import Path
import pandas as pd
import numpy as np
import psutil
import cv2
import torch

# Initialize DataFrame to store results
Memory_Analysis = pd.DataFrame(
    columns=['Frame', 'Object', 'Confidence', 'Total Memory (MB)', 'Used Memory (MB)', 'Free Memory (MB)',
             'Memory Usage (%)'])

# Dictionary to map class IDs to object names
class_names = {
    0: 'Metal Can',
    1: 'Water Bottle',
    # Add other class IDs and names as needed
}


def save_to_excel(df):
    """Save DataFrame to Excel file"""
    filename = 'Detection_Results(Windows_Nivida_GPU).xlsx'
    df.to_excel(filename, index=False, engine='openpyxl')
    print(f"Detection results saved to {filename}")


def analyze_frames(frames_folder, output_dir="detection_results"):
    """
    Analyze each frame in a folder using your custom YOLO model.

    Args:
        frames_folder (str): Path to folder containing frames
        output_dir (str): Output directory for visual results
    """
    # Check CUDA availability
    if torch.cuda.is_available():
        print(f"CUDA is available. Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("CUDA is not available. Using CPU.")
        
    global Memory_Analysis

    # Load your custom model
    data = r"/ultralytics/ultralytics/temp/Final_Version_Model.pt"
    model = YOLO(data)

    # Create output directory if it doesn't exist
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Get all image files in the folder
    image_files = [
        f for f in os.listdir(frames_folder)
        if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff'))
    ]

    print(f"Found {len(image_files)} frames to process")

    # Process each frame
    for img_file in image_files:
        img_path = os.path.join(frames_folder, img_file)

        # Get current memory usage
        memory = psutil.virtual_memory()
        total = memory.total / (1024 * 1024)
        used = memory.used / (1024 * 1024)
        free = memory.free / (1024 * 1024)
        percent = memory.percent

        # Run YOLO prediction (disable auto-saving)
        results = model.predict(img_path, save=False)

        # Process detection results
        for result in results:
            # Get annotated image
            annotated_frame = result.plot()

            # Save annotated image directly to output directory
            output_path = os.path.join(output_dir, f"detected_{img_file}")
            cv2.imwrite(output_path, annotated_frame)

            if result.boxes:
                labels = result.boxes.cls.cpu().numpy()
                confidences = result.boxes.conf.cpu().numpy()

                for label, confidence in zip(labels, confidences):
                    class_id = int(label)
                    object_name = class_names.get(class_id, f"Unknown_{class_id}")

                    # Add detection to DataFrame
                    new_row = pd.DataFrame({
                        'Frame': [img_file],
                        'Object': [object_name],
                        'Confidence': [confidence],
                        'Total Memory (MB)': [total],
                        'Used Memory (MB)': [used],
                        'Free Memory (MB)': [free],
                        'Memory Usage (%)': [percent]
                    })
                    Memory_Analysis = pd.concat([Memory_Analysis, new_row], ignore_index=True)

        print(f"Processed {img_file}: Detected {len(result.boxes) if result.boxes else 0} objects")

    # Save results to Excel after processing all frames
    save_to_excel(Memory_Analysis)
    print(f"Analysis complete. Visual results saved to {output_dir}")


# Example usage
if __name__ == "__main__":
    analyze_frames(
        frames_folder=r"/ultralytics/ultralytics/temp",
        output_dir="detection_results"
    )
