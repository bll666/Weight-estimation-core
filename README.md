# Weight-estimation-core
Key Code of the paper
Welcome to the Weght Estimation Core Module Project! This project utilizes a MobileNet V3 backbone with a PoseNet head to perform real-time keypoint detection on images. It includes functionalities for scale alignment and fusion using RANSAC and Bicubic interpolation.

## Table of Contents

1. [Overview](#overview)
2. [Prerequisites](#prerequisites)
3. [Installation](#installation)
4. [Usage](#usage)
5. [File Structure](#file-structure)


## Overview

This project aims to detect keypoints in images using a deep learning model optimized for mobile devices. The system includes:
- A MobilePoseNet model for keypoint detection.
- Scale alignment using RANSAC for projection transformation.
- Bicubic interpolation for smooth transition between scales.
- Saving and visualizing the detected keypoints.

## Prerequisites

To run this project, you need the following software and libraries:

- **Operating System**: Windows 10, macOS Catalina 10.15, or higher versions.
- **Python Version**: Python 3.8 or higher.
- **Hardware Requirements**: At least 8GB of RAM and a multi-core processor for optimal performance.
- **Python Libraries**: 
  - `torch`
  - `torchvision`
  - `opencv-python`
  - `numpy`

## Installation

To set up the project, follow these steps:

1. Clone the repository:
   ```
   git clone https://github.com/yourusername/mobileposenet-keypoint-detection.git
   ```

2. Navigate into the project directory:
   ```
   cd Weight_estimation_core

   ```

3. Create a virtual environment (optional but recommended):
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   ```

4. Install the required Python packages:
   ```
   pip install -r requirements.txt
   ```

## Usage

To run the keypoint detection system, execute the following command in your terminal:

```
python main.py
```

This will process the images in the specified directory, perform keypoint detection, scale alignment, and fusion, and save the results.

## File Structure

```
Weight_estimation_core/
│
├── mobileposenet.py           # The MobilePoseNet model definition
├── main.py                   # The main script for keypoint detection and processing
├── requirements.txt          # List of Python dependencies
├── README.md                 # This file
└── data/                     # Directory containing input images and saved keypoints
```

