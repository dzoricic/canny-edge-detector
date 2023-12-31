#include "canny_cpu.h"

using namespace std;

namespace CannyCPU {
    vector<int> gaussianBlur(int *pixels, vector<vector<double>> &kernel, double kernelConst, int sizeRows, int sizeCols, int sizeDepth)
    {
        vector<int> pixelsBlur(sizeRows * sizeCols * sizeDepth);
        for (int i = 0; i < sizeRows; i++)
        {
            for (int j = 0; j < sizeCols; j++)
            {
                for (int k = 0; k < sizeDepth; k++)
                {
                    double sum = 0;
                    double sumKernel = 0;
                    for (int y = -2; y <= 2; y++)
                    {
                        for (int x = -2; x <= 2; x++)
                        {
                            if ((i + x) >= 0 && (i + x) < sizeRows && (j + y) >= 0 && (j + y) < sizeCols)
                            {
                                double channel = (double)pixels[(i + x) * sizeCols * sizeDepth + (j + y) * sizeDepth + k];
                                sum += channel * kernelConst * kernel[x + 2][y + 2];
                                sumKernel += kernelConst * kernel[x + 2][y + 2];
                            }
                        }
                    }
                    pixelsBlur[i * sizeCols * sizeDepth + j * sizeDepth + k] = (int)(sum / sumKernel);
                }
            }
        }
        return pixelsBlur;
    }

    vector<int> rgbToGrayscale(vector<int> &pixels, int sizeRows, int sizeCols, int sizeDepth)
    {
        vector<int> pixelsGray(sizeRows * sizeCols);
        for (int i = 0; i < sizeRows; i++)
        {
            for (int j = 0; j < sizeCols; j++)
            {
                int sum = 0;
                for (int k = 0; k < sizeDepth; k++)
                {
                    sum = sum + pixels[i * sizeCols * sizeDepth + j * sizeDepth + k];
                }
                pixelsGray[i * sizeCols + j] = (int)(sum / sizeDepth);
            }
        }
        return pixelsGray;
    }

    void performNonMaximumSuppresion(double *G, std::vector<int> &theta, int sizeCols, int sizeRows, std::vector<int> &pixels, double largestG)
    {
        for (int i = 1; i < sizeRows - 1; i++)
        {
            for (int j = 1; j < sizeCols - 1; j++)
            {
                if (theta[i * sizeCols + j] == 0 || theta[i * sizeCols + j] == 180)
                {
                    if (G[i * sizeCols + j] < G[i * sizeCols + j - 1] || G[i * sizeCols + j] < G[i * sizeCols + j + 1])
                    {
                        G[i * sizeCols + j] = 0;
                    }
                }
                else if (theta[i * sizeCols + j] == 45 || theta[i * sizeCols + j] == 225)
                {
                    if (G[i * sizeCols + j] < G[(i + 1) * sizeCols + j + 1] || G[i * sizeCols + j] < G[(i - 1) * sizeCols + j - 1])
                    {
                        G[i * sizeCols + j] = 0;
                    }
                }
                else if (theta[i * sizeCols + j] == 90 || theta[i * sizeCols + j] == 270)
                {
                    if (G[i * sizeCols + j] < G[(i + 1) * sizeCols + j] || G[i * sizeCols + j] < G[(i - 1) * sizeCols + j])
                    {
                        G[i * sizeCols + j] = 0;
                    }
                }
                else
                {
                    if (G[i * sizeCols + j] < G[(i + 1) * sizeCols + j - 1] || G[i * sizeCols + j] < G[(i - 1) * sizeCols + j + 1])
                    {
                        G[i * sizeCols + j] = 0;
                    }
                }

                pixels[i * sizeCols + j] = (int)(G[i * sizeCols + j] * (255.0 / largestG));
            }
        }
    };

    void performDoubleThresholding(double *G, vector<int> &theta, int sizeCols, int sizeRows, vector<int> &pixels, double largestG)
    {
        double lowerThreshold = 0.03;
        double higherThreshold = 0.1;
        bool changes;
        do
        {
            changes = false;
            for (int i = 1; i < sizeRows - 1; i++)
            {
                for (int j = 1; j < sizeCols - 1; j++)
                {
                    if (G[i * sizeCols + j] < (lowerThreshold * largestG))
                    {
                        G[i * sizeCols + j] = 0;
                    }
                    else if (G[i * sizeCols + j] >= (higherThreshold * largestG))
                    {
                        continue;
                    }
                    else if (G[i * sizeCols + j] < (higherThreshold * largestG))
                    {
                        G[i * sizeCols + j] = 0;
                        for (int x = -1; x <= 1; x++)
                        {
                            bool breakNestedLoop = false;
                            for (int y = -1; y <= 1; y++)
                            {
                                if (x == 0 && y == 0)
                                {
                                    continue;
                                }
                                if (G[(i + x) * sizeCols + (j + y)] >= (higherThreshold * largestG))
                                {
                                    G[i * sizeCols + j] = (higherThreshold * largestG);
                                    changes = true;
                                    breakNestedLoop = true;
                                    break;
                                }
                            }
                            if (breakNestedLoop)
                            {
                                break;
                            }
                        }
                    }
                    pixels[i * sizeCols + j] = (int)(G[i * sizeCols + j] * (255.0 / largestG));
                }
            }
        } while (changes);
    }
    vector<int> cannyFilter(vector<int> &pixels, int sizeRows, int sizeCols, int sizeDepth)
    {
        vector<int> pixelsCanny(sizeRows * sizeCols);
        int gx[3][3] = {{-1, 0, 1}, {-2, 0, 2}, {-1, 0, 1}};
        int gy[3][3] = {{-1, -2, -1}, {0, 0, 0}, {1, 2, 1}};
        double *G = new double[sizeRows * sizeCols];
        std::vector<int> theta(sizeRows * sizeCols);
        double largestG = 0;

        // perform canny edge detection on everything but the edges
        for (int i = 1; i < sizeRows - 1; i++)
        {
            for (int j = 1; j < sizeCols - 1; j++)
            {
                // find gx and gy for each pixel
                double gxValue = 0;
                double gyValue = 0;
                for (int x = -1; x <= 1; x++)
                {
                    for (int y = -1; y <= 1; y++)
                    {
                        gxValue = gxValue + (gx[1 - x][1 - y] * (double)(pixels[(i + x) * sizeCols + j + y]));
                        gyValue = gyValue + (gy[1 - x][1 - y] * (double)(pixels[(i + x) * sizeCols + j + y]));
                    }
                }

                // calculate G and theta
                G[i * sizeCols + j] = std::sqrt(std::pow(gxValue, 2) + std::pow(gyValue, 2));
                double atanResult = atan2(gyValue, gxValue) * 180.0 / 3.14159265;
                theta[i * sizeCols + j] = (int)(180.0 + atanResult);

                if (G[i * sizeCols + j] > largestG)
                {
                    largestG = G[i * sizeCols + j];
                }

                // setting the edges
                if (i == 1)
                {
                    G[i * sizeCols + j - 1] = G[i * sizeCols + j];
                    theta[i * sizeCols + j - 1] = theta[i * sizeCols + j];
                }
                else if (j == 1)
                {
                    G[(i - 1) * sizeCols + j] = G[i * sizeCols + j];
                    theta[(i - 1) * sizeCols + j] = theta[i * sizeCols + j];
                }
                else if (i == sizeRows - 1)
                {
                    G[i * sizeCols + j + 1] = G[i * sizeCols + j];
                    theta[i * sizeCols + j + 1] = theta[i * sizeCols + j];
                }
                else if (j == sizeCols - 1)
                {
                    G[(i + 1) * sizeCols + j] = G[i * sizeCols + j];
                    theta[(i + 1) * sizeCols + j] = theta[i * sizeCols + j];
                }

                // setting the corners
                if (i == 1 && j == 1)
                {
                    G[(i - 1) * sizeCols + j - 1] = G[i * sizeCols + j];
                    theta[(i - 1) * sizeCols + j - 1] = theta[i * sizeCols + j];
                }
                else if (i == 1 && j == sizeCols - 1)
                {
                    G[(i - 1) * sizeCols + j + 1] = G[i * sizeCols + j];
                    theta[(i - 1) * sizeCols + j + 1] = theta[i * sizeCols + j];
                }
                else if (i == sizeRows - 1 && j == 1)
                {
                    G[(i + 1) * sizeCols + j - 1] = G[i * sizeCols + j];
                    theta[(i + 1) * sizeCols + j - 1] = theta[i * sizeCols + j];
                }
                else if (i == sizeRows - 1 && j == sizeCols - 1)
                {
                    G[(i + 1) * sizeCols + j + 1] = G[i * sizeCols + j];
                    theta[(i + 1) * sizeCols + j + 1] = theta[i * sizeCols + j];
                }

                // round to the nearest 45 degrees
                theta[i * sizeCols + j] = round(theta[i * sizeCols + j] / 45) * 45;
            }
        }

        performNonMaximumSuppresion(G, theta, sizeCols, sizeRows, pixelsCanny, largestG);

        performDoubleThresholding(G, theta, sizeCols, sizeRows, pixelsCanny, largestG);

        return pixelsCanny;
    };

    void cannyEdgeDetection(uint8_t *inputImage, int width, int height, int channels, char *outputImagePath)
    {
        int *pixels = imgToArray(inputImage, height, width, channels);

        // GAUSSIAN_BLUR:

        vector<vector<double>> kernel = {{2.0, 4.0, 5.0, 4.0, 2.0},
                                        {4.0, 9.0, 12.0, 9.0, 4.0},
                                        {5.0, 12.0, 15.0, 12.0, 5.0},
                                        {4.0, 9.0, 12.0, 9.0, 4.0},
                                        {2.0, 4.0, 5.0, 4.0, 2.0}};
        double kernelConst = (1.0 / 159.0);
        vector<int> pixelsBlur = gaussianBlur(pixels, kernel, kernelConst, height, width, channels);

        // GRAYSCALE:

        vector<int> pixelsGray = rgbToGrayscale(pixelsBlur, height, width, channels);

        // CANNY_FILTER:

        vector<int> pixelsCanny = cannyFilter(pixelsGray, height, width, 1);
        int* pixelCannyArray = pixelsCanny.data();

        uint8_t *outputImage = (unsigned char *)malloc(width * height);
        arrayToImg(pixelCannyArray, outputImage, height, width, 1);

        stbi_write_jpg(outputImagePath, width, height, 1, outputImage, 100);

        free(outputImage);
    }
}
