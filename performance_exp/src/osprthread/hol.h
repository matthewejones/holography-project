#pragma once

#include <opencv2/opencv.hpp>
#include <fftw3.h>
#include <random>

const int N = 8;
const double pi = 3.141592654;

void fillRandom(cv::Mat& image);
void ospr(cv::Mat& I, cv::Mat& H);
void ospr(cv::Mat& I, cv::Mat& H, bool Threading);
void sub_ospr(cv::Mat& I, cv::Mat& H, int i, int iteration, fftw_plan& plan, std::mt19937_64& gen, std::uniform_real_distribution<double>& dis);
void ifftshift(fftw_complex* out, const fftw_complex* in, int xdim, int ydim);
void fillConsecutive(cv::Mat& I);
void printMat(const fftw_complex* in, int width, int height);
void fillConsecutive(fftw_complex* I, int width, int height);
void printMat(cv::Mat& I);