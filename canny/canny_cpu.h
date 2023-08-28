#ifndef CANNY_CPU_H
#define CANNY_CPU_H
using namespace std;
#include <iostream>
#include <vector>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <chrono>
#include "../utils/image_handler.h"
#include "../utils/stb_image.h"
#include "../utils/stb_image_write.h"

namespace CannyCPU {
  extern vector<int> gaussianBlur(int *pixels, vector<vector<double>> &kernel, double kernelConst, int sizeRows, int sizeCols, int sizeDepth);
  extern vector<int> rgbToGrayscale(vector<int> &pixels, int sizeRows, int sizeCols, int sizeDepth);
  extern void performNonMaximumSuppresion(double *G, vector<int> &theta, int sizeCols, int sizeRows, vector<int> &pixels, double largestG);
  extern void performDoubleThresholding(double *G, vector<int> &theta, int sizeCols, int sizeRows, vector<int> &pixels, double largestG);
  extern vector<int> cannyFilter(vector<int> &pixels, int sizeRows, int sizeCols, int sizeDepth);
  extern void cannyEdgeDetection(uint8_t *inputImage, int width, int height, int channels, char *outputImagePath);
}

#endif