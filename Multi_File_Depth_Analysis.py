#!/usr/bin/env python3

import sys
import os
import numpy as np
from matplotlib import pyplot as plt
import pandas as pd

from jetson_inference import depthNet
from jetson_utils import loadImage, saveImage
from depthnet_utils import depthBuffers

# =========================================
# CONFIGURATION
# =========================================
INPUT_FOLDER = r"/home/jetson/masked_objects-20250620T142956Z-1-001/masked_objects"  # Folder with input images
OUTPUT_FOLDER = r"/home/jetson/output"  # Where to save results
NETWORK = "fcn-mobilenet"
VISUALIZATION = "depth"
COLORMAP = "viridis-inverted"
# =========================================

# Initialize DataFrame to store results
Depth_Results = pd.DataFrame(columns=['File Name', 'Resolution', 'Min depth (m)', 'Max depth (m)', 'Mean depth (m)'])

# Create output folder if it doesn't exist
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# Load depth estimation network
net = depthNet(NETWORK)

def save_to_excel(df):
    """Save DataFrame to Excel file"""
    filename = 'Depth_Results.xlsx'
    df.to_excel(filename, index=False, engine='openpyxl')
    print(f"Detection results saved to {filename}")


# Helper class for depth buffer args
class Args:
    def __init__(self):
        self.visualize = VISUALIZATION
        self.depth_size = 1.0


# Loop through all image files
valid_extensions = [".jpg", ".jpeg", ".png"]
for filename in os.listdir(INPUT_FOLDER):
    if not any(filename.lower().endswith(ext) for ext in valid_extensions):
        continue

    image_path = os.path.join(INPUT_FOLDER, filename)
    base_filename = os.path.splitext(filename)[0]

    print(f"\nProcessing {image_path}...")

    if not os.path.isfile(image_path):
        print(f"File {filename} not found, skipping.")
        continue

    # Load input image
    img_input = loadImage(image_path)

    # Create buffer manager
    buffers = depthBuffers(Args())
    buffers.Alloc(img_input.shape, img_input.format)

    # Process image to generate depth
    net.Process(img_input, buffers.depth, COLORMAP)

    # Get raw depth data
    depth_field = net.GetDepthField()
    depth_width = net.GetDepthFieldWidth()
    depth_height = net.GetDepthFieldHeight()
    depth_map = np.array(depth_field, dtype=np.float32).reshape((depth_height, depth_width))

    # Save raw depth map
    raw_output_path = os.path.join(OUTPUT_FOLDER, f"{base_filename}_depth.npy")
    np.save(raw_output_path, depth_map)
    print(f"Saved raw depth data to {raw_output_path}")

    # Analyze depths
    valid_depths = depth_map[depth_map > 0]
    if valid_depths.size == 0:
        print("No valid depth values found, skipping analysis.")
        continue

    min_depth = np.min(valid_depths)
    max_depth = np.max(valid_depths)
    mean_depth = np.mean(valid_depths)
    Resolution = depth_width * depth_height

    print("Depth Field Analysis:")
    print(f"- Resolution: {depth_width}x{depth_height}")
    print(f"- Min depth: {min_depth:.2f} meters")
    print(f"- Max depth: {max_depth:.2f} meters")
    print(f"- Mean depth: {mean_depth:.2f} meters")

    # Save histogram
    hist_path = os.path.join(OUTPUT_FOLDER, f"{base_filename}_hist.png")
    plt.figure(figsize=(10, 5))
    plt.hist(valid_depths.flatten(), bins=50, range=(0, max_depth))
    plt.xlabel('Depth (meters)')
    plt.ylabel('Pixel Count')
    plt.title(f'Depth Distribution - {filename}')
    plt.savefig(hist_path)
    plt.close()
    print(f"Saved depth histogram to {hist_path}")

    # Save depth visualization image
    vis_output_path = os.path.join(OUTPUT_FOLDER, f"{base_filename}_vis.jpg")
    saveImage(vis_output_path, buffers.depth)
    print(f"Saved depth visualization to {vis_output_path}")

    # Adding data to excel sheet
    new_row = pd.DataFrame({
        'File Name': [filename],
        'Resolution': [Resolution],
        'Min depth (m)': [min_depth],
        'Max depth (m)': [max_depth],
        'Mean depth (m)': [mean_depth]
    })
    Depth_Results = pd.concat([Depth_Results, new_row], ignore_index=True)

# Saves the Excel table to the folder
save_to_excel(Depth_Results)