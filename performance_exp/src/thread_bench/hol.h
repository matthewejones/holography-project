#pragma once

#include <opencv2/opencv.hpp>
#include <fftw3.h>

const int N = 8;
const double pi = 3.141592654;

void fillRandom(cv::Mat& image);
void ospr(cv::Mat& I, cv::Mat& H, fftw_plan &plan);
void ifftshift(fftw_complex* out, const fftw_complex* in, int xdim, int ydim);
void fillConsecutive(cv::Mat& I);
void printMat(const fftw_complex* in, int width, int height);
void fillConsecutive(fftw_complex* I, int width, int height);
void printMat(cv::Mat& I);