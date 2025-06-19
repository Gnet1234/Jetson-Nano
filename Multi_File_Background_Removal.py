import cv2
import numpy as np
from ultralytics import YOLO
import os

# Configuration
input_folder = r"C:\Users\garne\OneDrive\Documents\Extracted_Frames"
output_folder = "masked_objects"
Model_Path = r"C:\Users\garne\Downloads\Final_Version_Model.pt"

# Create output folder if it doesn't exist
os.makedirs(output_folder, exist_ok=True)

# Load YOLOv8 segmentation model
model_yolo = YOLO(Model_Path)

# Supported image extensions
valid_extensions = [".jpg", ".jpeg", ".png", ".bmp"]

# Loop through each image in the input folder
for filename in os.listdir(input_folder):
    if any(filename.lower().endswith(ext) for ext in valid_extensions):
        image_path = os.path.join(input_folder, filename)
        print(f"Processing {image_path}...")

        # Run YOLO segmentation
        results = model_yolo(image_path)

        if results[0].masks is None or len(results[0].masks) == 0:
            print(f"No masks found for {filename}, skipping.")
            continue

        # Take the first mask
        mask = results[0].masks[0].data[0].numpy()

        # Read and resize mask to match image
        image = cv2.imread(image_path)
        mask_resized = cv2.resize((mask * 255).astype(np.uint8), (image.shape[1], image.shape[0]))

        # Apply mask
        masked_image = cv2.bitwise_and(image, image, mask=mask_resized)

        # Save masked image
        output_path = os.path.join(output_folder, f"masked_{filename}")
        cv2.imwrite(output_path, masked_image)
        print(f"Saved masked image to: {output_path}")
