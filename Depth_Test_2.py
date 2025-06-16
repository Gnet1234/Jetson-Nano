#!/usr/bin/env python3
#
# Enhanced depthnet.py with depth field extraction
#

import sys
import os
import numpy as np
from matplotlib import pyplot as plt

from jetson_inference import depthNet
from jetson_utils import loadImage, saveImage, cudaAllocMapped
from depthnet_utils import depthBuffers

def main():
    # =========================================
    # CONFIGURATION - MODIFY THESE VALUES
    # =========================================
    INPUT_PATH = r"/home/jetson/Extracted_Frames/frame_0000.jpg" # Path to input image
    OUTPUT_PATH = r"/home/jetson/output/depth_test.jpg"  # Output depth visualization
    DEPTH_DATA_PATH = "depth_data.npy" # Raw depth data output
    NETWORK = "fcn-mobilenet"        # Model to use
    VISUALIZATION = "depth"          # "input", "depth", or "input,depth"
    COLORMAP = "viridis-inverted"    # Colormap for visualization
    # =========================================

    # Verify input file exists
    if not os.path.isfile(INPUT_PATH):
        print(f"Error: Input file not found at {INPUT_PATH}")
        sys.exit(1)

    # Load the depth network
    net = depthNet(NETWORK)

    # Load input image
    img_input = loadImage(INPUT_PATH)
    
    # Create buffer manager
    class Args:
        def __init__(self):
            self.visualize = VISUALIZATION
            self.depth_size = 1.0
            
    buffers = depthBuffers(Args())
    buffers.Alloc(img_input.shape, img_input.format)

    # Process the image to get depth map
    net.Process(img_input, buffers.depth, COLORMAP)

    # =========================================
    # DEPTH FIELD EXTRACTION AND ANALYSIS
    # =========================================
    
    # Get the raw depth field (1D array)
    depth_field = net.GetDepthField()
    
    # Get depth map dimensions
    depth_width = net.GetDepthFieldWidth()
    depth_height = net.GetDepthFieldHeight()
    
    # Convert to 2D numpy array
    depth_map = np.array(depth_field, dtype=np.float32).reshape((depth_height, depth_width))
    
    # Save raw depth data
    np.save(DEPTH_DATA_PATH, depth_map)
    print(f"Saved raw depth data to {DEPTH_DATA_PATH}")
    
    # Calculate basic statistics
    valid_depths = depth_map[depth_map > 0]  # Filter out invalid depths
    min_depth = np.min(valid_depths)
    max_depth = np.max(valid_depths)
    mean_depth = np.mean(valid_depths)
    
    print("\nDepth Field Analysis:")
    print(f"- Resolution: {depth_width}x{depth_height}")
    print(f"- Min depth: {min_depth:.2f} meters")
    print(f"- Max depth: {max_depth:.2f} meters")
    print(f"- Mean depth: {mean_depth:.2f} meters")
    
    # Create depth histogram
    plt.figure(figsize=(10, 5))
    plt.hist(valid_depths.flatten(), bins=50, range=(0, max_depth))
    plt.xlabel('Depth (meters)')
    plt.ylabel('Pixel Count')
    plt.title('Depth Distribution Histogram')
    plt.savefig('depth_histogram.png')
    print("Saved depth histogram to depth_histogram.png")
    
    # =========================================
    # OUTPUT VISUALIZATION
    # =========================================
    
    # Save the depth visualization
    saveImage(OUTPUT_PATH, buffers.depth)
    print(f"\nSaved depth visualization to {OUTPUT_PATH}")

if __name__ == "__main__":
    main()
