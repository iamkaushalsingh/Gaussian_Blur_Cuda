# CUDA Gaussian Blur

## Overview
This project implements a Gaussian blur filter using CUDA for parallel processing. The application applies the Gaussian blur multiple times to an image, using a 3x3 Gaussian kernel. The main purpose of this project is to demonstrate the effective use of CUDA for image processing tasks, leveraging the parallel computation power of GPUs.

# Requirements
Operating System: Windows, Linux, or macOS
Compiler: NVIDIA CUDA Toolkit (nvcc)
Dependencies:
OpenCV 4.x (used for image loading and saving)
NVIDIA GPU with CUDA Compute Capability 3.5 or higher
Setup and Installation
Install CUDA Toolkit:

Download and install the appropriate version of the CUDA Toolkit from NVIDIA's official site based on your operating system.
Ensure that the nvcc compiler is in your system's PATH.

# Install OpenCV:
You can install OpenCV from source or use a pre-built package depending on your platform. Ensure that OpenCV is accessible in your environment.
Clone the Repository:

Clone this repository to your local machine using git clone, or download the source code as a zip file.

# Compilation
To compile the Gaussian Blur application, navigate to the directory containing the source code and run the following command: 

# nvcc -o gaussianBlur main.cu `pkg-config --cflags --libs opencv4`

This command compiles the CUDA code and links it with OpenCV libraries. Make sure pkg-config is set up correctly to point to your OpenCV installation.

# Usage
To run the Gaussian Blur application, execute the compiled binary with:

./gaussianBlur
The application reads an image file named Dude.jpg from the working directory, applies Gaussian blurring 100 times, and saves the result as Dude_blurred.jpg.

# Contributions
Contributions to this project are welcome. Please follow the standard fork-pull request workflow. Ensure you adhere to the coding standards and provide sufficient documentation and tests for your contributions.

# License
Specify your licensing terms here. Typically, this project would be under an MIT License or another permissive open-source license.
