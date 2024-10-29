import cv2
import numpy as np
import os

def gaussian_pyramid(img, levels, kernel_size=(5, 5)):

    gaussian_pyramid_levels = [img]
    for _ in range(1, levels):
        # Downsample the image using Gaussian filter and then downscale by 2
        img = cv2.GaussianBlur(gaussian_pyramid_levels[-1], kernel_size, 0)
        img = cv2.pyrDown(img)
        gaussian_pyramid_levels.append(img)
    return gaussian_pyramid_levels

# Define the path to the 'weight' folder on D drive
dataset_path = 'D:/weight/dataset1.0'

# Check if the directory exists
if not os.path.exists(dataset_path):
    print(f"Error: Directory {dataset_path} does not exist.")
else:
    
    image_files = [f for f in os.listdir(dataset_path) if f.lower().endswith('.png')]

    # Process each PNG image in the directory
    for image_file in image_files:
        image_path = os.path.join(dataset_path, image_file)
        dataset = cv2.imread(image_path)

        # Ensure the image is loaded correctly
        if dataset is None:
            print(f"Error: Image not found or cannot be read at {image_path}.")
            continue

        # Create a 3-level Gaussian pyramid
        levels = 3
        pyramid_levels = gaussian_pyramid(dataset, levels)

        # Save or process the pyramid levels
        for i, level in enumerate(pyramid_levels):
            output_path = os.path.join(dataset_path, f'pyramid_level_{i}_{image_file}')
            cv2.imwrite(output_path, level)
            print(f'Saved level {i} of {image_file} to {output_path}')

        print(f"Processed {image_file}")
