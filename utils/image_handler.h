#ifndef IMAGE_HANDLER_H
#define IMAGE_HANDLER_H

#include <iostream>
#include <vector>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <algorithm>

using namespace std;

extern int *imgToArray(uint8_t *pixelPtr, int sizeRows, int sizeCols, int sizeDepth);
extern void arrayToImg(int *pixels, uint8_t *pixelPtr, int sizeRows, int sizeCols, int sizeDepth);

#endif