#include <iostream>
#include <vector>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <algorithm>
#include <chrono>
#include "canny/canny_cpu.h"
#include "canny/canny_gpu.h"
#include "canny/canny_shared_memory.h"
#include "utils/image_handler.h"

#define STB_IMAGE_IMPLEMENTATION
#include "utils/stb_image.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "utils/stb_image_write.h"

#define MAX_NAME_LENGTH (64)

using namespace std;
using namespace std::chrono;

int main(int argc, char* argv[])
{
    if (argc != 2) {
        cout << "Usage: " << argv[0] << " <input_image_path>" << endl;
        return 1;
    }

    int width, height, channels;
    double lowerThreshold = 0.03;
    double higherThreshold = 0.1;

    char *inputImagePath = argv[1];
    char outputImageCPU[MAX_NAME_LENGTH] = "output_cpu.jpg";
    char outputImageGPU[MAX_NAME_LENGTH] = "output_gpu.jpg";
    char outputImageSM[MAX_NAME_LENGTH] = "output_gpu_sm.jpg";
    uint8_t *inputImage = stbi_load(inputImagePath, &width, &height, &channels, STBI_rgb);
    
    if (inputImage == NULL)
    {
        printf("Failed to load image: %s\n", inputImagePath);
        return -1;
    }

    if (channels != 3)
    {
        printf("Image is not in RGB format\n");
        return -1;
    }

    // Running on CPU

    auto start_cpu = high_resolution_clock::now();

    CannyCPU::cannyEdgeDetection(inputImage, width, height, channels, outputImageCPU);

    auto stop_cpu = high_resolution_clock::now();

    auto duration_cpu = duration_cast<milliseconds>(stop_cpu - start_cpu);
    cout << "CPU time: " << duration_cpu.count() << " ms" << endl;

    // Running on GPU

    auto start_gpu = high_resolution_clock::now();

    CannyGPU::cannyEdgeDetection(inputImage, lowerThreshold, higherThreshold, width, height, channels, outputImageGPU);

    auto stop_gpu = high_resolution_clock::now();

    auto duration_gpu = duration_cast<milliseconds>(stop_gpu - start_gpu);
    cout << "GPU time: " << duration_gpu.count() << " ms" << endl;

    // Running on GPU with shared memory

    auto start_sm = high_resolution_clock::now();

    CannySM::cannyEdgeDetection(inputImage, lowerThreshold, higherThreshold, width, height, channels, outputImageSM);

    auto stop_sm = high_resolution_clock::now();

    auto duration_sm = duration_cast<milliseconds>(stop_sm - start_sm);
    cout << "GPU (shared memory) time: " << duration_sm.count() << " ms" << endl;

    return 0;
}