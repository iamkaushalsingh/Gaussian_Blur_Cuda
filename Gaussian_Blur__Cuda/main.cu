#include <stdio.h>
#include <stdlib.h>
#include <opencv2/opencv.hpp>

#define MASK_WIDTH 3
#define TILE_WIDTH 16
#define CHANNELS 3
#define ITERATIONS 100 

_constant_ float d_mask[MASK_WIDTH][MASK_WIDTH] = {
    {1, 2, 1},
    {2, 4, 2},
    {1, 2, 1}
};

_global_ void gaussianBlur(unsigned char *inputImage, unsigned char *outputImage, int width, int height) {
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int row_o = blockIdx.y * blockDim.y + ty;
    int col_o = blockIdx.x * blockDim.x + tx;

    float result[CHANNELS] = {0};

    if (row_o < height && col_o < width) {
        for (int i = -MASK_WIDTH/2; i <= MASK_WIDTH/2; ++i) {
            for (int j = -MASK_WIDTH/2; j <= MASK_WIDTH/2; ++j) {
                int row_i = min(max(row_o + i, 0), height - 1);
                int col_i = min(max(col_o + j, 0), width - 1);

                for (int c = 0; c < CHANNELS; ++c) {
                    result[c] += inputImage[(row_i * width + col_i) * CHANNELS + c] * d_mask[i + MASK_WIDTH/2][j + MASK_WIDTH/2];
                }
            }
        }

        for (int c = 0; c < CHANNELS; ++c) {
            outputImage[(row_o * width + col_o) * CHANNELS + c] = (unsigned char)(result[c] / 16.0f);
        }
    }
}

int main() {
    
    cv::Mat inputImage = cv::imread("Dude.jpg", cv::IMREAD_COLOR);

    if (inputImage.empty()) {
        printf("Error: Unable to load image.\n");
        return -1;
    }

    int width = inputImage.cols;
    int height = inputImage.rows;

    // Allocate memory for input and output images on GPU
    size_t imageSize = width * height * CHANNELS * sizeof(unsigned char);
    unsigned char *d_inputImage, *d_outputImage;
    cudaMalloc(&d_inputImage, imageSize);
    cudaMalloc(&d_outputImage, imageSize);

    // Copy input image to GPU memory
    cudaMemcpy(d_inputImage, inputImage.data, imageSize, cudaMemcpyHostToDevice);

    // Define grid and block dimensions for CUDA kernel
    dim3 threadsPerBlock(TILE_WIDTH, TILE_WIDTH);
    dim3 numBlocks((width + TILE_WIDTH - 1) / TILE_WIDTH, (height + TILE_WIDTH - 1) / TILE_WIDTH);

    // Apply multiple iterations of Gaussian blur
    for (int i = 0; i < ITERATIONS; ++i) {
        gaussianBlur<<<numBlocks, threadsPerBlock>>>(d_inputImage, d_outputImage, width, height);
        std::swap(d_inputImage, d_outputImage); // Swap input and output for next iteration
    }

    // Copy final output image from GPU to CPU
    cudaMemcpy(inputImage.data, d_inputImage, imageSize, cudaMemcpyDeviceToHost);

    // Save the output image
    cv::imwrite("Dude_blurred.jpg", inputImage);

    // Free GPU memory
    cudaFree(d_inputImage);
    cudaFree(d_outputImage);

    return 0;
}