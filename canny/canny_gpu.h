#ifndef CANNY_GPU_H
#define CANNY_GPU_H
#include "../utils/image_handler.h"
#include <cuda_runtime.h>

#include "../utils/stb_image.h"
#include "../utils/stb_image_write.h"

namespace CannyGPU {
  extern __global__ void gaussianBlur(int *inputPixels, int *blurredPixels, int sizeRows, int sizeCols, int sizeDepth);
  extern __global__ void rgbToGrayscale(int *blurredPixels, int *grayscaledPixels, int sizeRows, int sizeCols, int sizeDepth);
  extern __global__ void nonMaxSuppresion(int *theta, double *G, int sizeRows, int sizeCols, double largestG, int *pixelsCanny);
  extern __global__ void cannyFilter(int *grayscaledPixels, int sizeRows, int sizeCols, double *G, int *theta);
  extern void doubleThreshold(int sizeRows, int sizeCols, double *G, double largestG, int *pixelsCanny);
  extern double comp(double a, double b);
  extern void cannyEdgeDetection(uint8_t *inputImage, double lowerThreshold, double higherThreshold, int width, int height, int channels, char *outputImagePath);
}

#endif