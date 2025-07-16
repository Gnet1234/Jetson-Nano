#!/usr/bin/env python3

import os
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import pandas as pd
from torchvision import transforms
from train_depthnet import DepthNet 

INPUT_SIZE = (1280, 720)  # Must match training input
INPUT_FOLDER = r"C:\Users\Admin\Documents\Depthnet_Tranining_Flies\masked_objects"

# Device
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def Image_Analysis(OUTPUT_FOLDER, MODEL_PATH):
        # ========== SETUP ==========
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)

    # Load model
    model = DepthNet().to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()
    print(f"Loaded model from {MODEL_PATH} onto {DEVICE}.")

    # Image transform
    transform = transforms.Compose([
        transforms.Resize(INPUT_SIZE),
        transforms.ToTensor(),
    ])

    # Prepare results table
    results = pd.DataFrame(columns=['File Name', 'Resolution', 'Min depth (m)', 'Max depth (m)', 'Mean depth (m)'])

    # Loop through images
    valid_extensions = [".jpg", ".jpeg", ".png"]
    for filename in os.listdir(INPUT_FOLDER):
        if not any(filename.lower().endswith(ext) for ext in valid_extensions):
            continue

        image_path = os.path.join(INPUT_FOLDER, filename)
        base_filename = os.path.splitext(filename)[0]

        print(f"\nProcessing: {filename}")

        # Load and preprocess
        image = Image.open(image_path).convert("RGB")
        input_tensor = transform(image).unsqueeze(0).to(DEVICE)

        # Run inference
        with torch.no_grad():
            output = model(input_tensor)  # Shape: [1, 1, H, W]
        depth_map = output.squeeze().cpu().numpy()  # Shape: [H, W]

        # Save raw depth data
        npy_path = os.path.join(OUTPUT_FOLDER, f"{base_filename}_depth.npy")
        np.save(npy_path, depth_map)
        print(f"Saved raw depth to {npy_path}")

        # Analyze
        valid_depths = depth_map[depth_map > 0]
        if valid_depths.size == 0:
            print("No valid depth values found, skipping.")
            continue

        min_depth = np.min(valid_depths)
        max_depth = np.max(valid_depths)
        mean_depth = np.mean(valid_depths)
        resolution = depth_map.shape[0] * depth_map.shape[1]

        print(f"→ Resolution: {depth_map.shape[::-1]}")
        print(f"→ Min: {min_depth:.2f} | Max: {max_depth:.2f} | Mean: {mean_depth:.2f}")

        # Save histogram
        hist_path = os.path.join(OUTPUT_FOLDER, f"{base_filename}_hist.png")
        plt.figure(figsize=(10, 5))
        plt.hist(valid_depths.flatten(), bins=50, range=(0, max_depth))
        plt.xlabel('Depth (meters)')
        plt.ylabel('Pixel Count')
        plt.title(f'Depth Distribution - {filename}')
        plt.tight_layout()
        plt.savefig(hist_path)
        plt.close()
        print(f"Saved histogram to {hist_path}")

        # Save visual image
        vis_path = os.path.join(OUTPUT_FOLDER, f"{base_filename}_vis.png")
        depth_vis = (depth_map - depth_map.min()) / (depth_map.max() - depth_map.min() + 1e-8)
        plt.imsave(vis_path, depth_vis, cmap="viridis")
        print(f"Saved visualization to {vis_path}")

        # Append to results
        results = pd.concat([results, pd.DataFrame({
            'File Name': [filename],
            'Resolution': [resolution],
            'Min depth (m)': [min_depth],
            'Max depth (m)': [max_depth],
            'Mean depth (m)': [mean_depth]
        })], ignore_index=True)

    # Save Excel
    excel_path = os.path.join(OUTPUT_FOLDER, "Depth_Results.xlsx")
    results.to_excel(excel_path, index=False)
    print(f"\n✅ Saved depth analysis results to {excel_path}")


for i in range(10,100,10):
    Output_Folder = rf"C:\Users\Admin\Documents\Depthnet_Tranining_Flies\tiny_depthnet_40_Epochs_{i}%"
    Model_Path = rf"C:\Users\Admin\Documents\Depthnet_Tranining_Flies\tiny_depthnet_tensor_40_{i}%.pth"
    Image_Analysis(OUTPUT_FOLDER=Output_Folder, MODEL_PATH= Model_Path)
    # print(Output_Folder)
    # print(Model_Path)
