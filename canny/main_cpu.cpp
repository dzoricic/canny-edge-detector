#include <iostream>
#include <vector>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <chrono>
#include "canny_cpu.h"

#define STB_IMAGE_IMPLEMENTATION
#include "../utils/stb_image.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "../utils/stb_image_write.h"

using namespace std;
using namespace std::chrono;

#define MAX_FILE_NAME (64)

int main()
{
    int width, height, channels;
    char inputImagePath[MAX_FILE_NAME] = "input.jpg";
    char outputImagePath[MAX_FILE_NAME] = "output.jpg";

    // Load the input image
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

    auto start = high_resolution_clock::now();

    CannyCPU::cannyEdgeDetection(inputImage, width, height, channels, outputImagePath);

    auto stop = high_resolution_clock::now();

    auto duration = duration_cast<milliseconds>(stop - start);
    cout << "Time of execution on CPU is: " << duration.count() << " ms" << endl;

    return 0;
}