#include "hol.h"

#include <ctime>
#include <opencv2/imgcodecs.hpp>
#include <complex>
#include <cstdlib>
#include <string>
#include <cmath>
#include <iostream>
#include <functional>


/*** Return a hologram of the same size as the image ***/
void ospr(cv::Mat &I, cv::Mat &H)
{
    fftw_complex* E = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * I.cols * I.rows);
    fftw_complex* Eshift = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * I.cols * I.rows);
    uchar *p;
    fftw_plan plan = fftw_plan_dft_2d(I.rows, I.cols, Eshift, Eshift, FFTW_BACKWARD, 0);
    std::mt19937_64 gen(123);
    std::uniform_real_distribution<double> dis(0.0, 2 * pi);

    double phi;
    std::complex<double> hpixel;


    for (int i = 0; i < 3; i++)
    {
        // Fill with zeros

        for (int iteration = 0; iteration < N; iteration++)
        {

            /// We can split this into a thread
            for (std::size_t y = 0; y < I.rows; y++)
            {
                
                p = I.ptr<uchar>(y);
                for (std::size_t x = 0; x < I.cols; x++)
                {
                    //Random Phase
                    phi = dis(gen);
                    hpixel = std::polar(sqrt(p[x]), phi);
                    // Multiply by square root of image intensity
                    E[y * I.cols + x][0] = hpixel.real();
                    E[y * I.cols + x][1] = hpixel.imag();
                }
            }
            //printMat(E, WIDTH, HEIGHT);
            // Next we perfofrm the inverse fourier transform of this
            ifftshift(Eshift, E, I.cols, I.rows);

            //printMat(Eshift, WIDTH, HEIGHT);

            fftw_execute(plan);
            

            
            // Maybe this is the point where it fails?
            for (std::size_t y = 0; y < I.rows; y++)
            {
                p = H.ptr<uchar>(y);
                for (std::size_t x = 0; x < I.cols; x++)
                {
                    // This can be optimised
                    if (Eshift[y * I.cols + x][1] > 0 || (Eshift[y * I.cols + x][1] == 0 && Eshift[y * I.cols + x][0] > 0))
                    {
                        p[3*x + i] |= (1 << iteration);
                    }
                    else
                    {
                        p[3*x + i] &= ~(1 << iteration);
                    }
                }
            }
            
        }
    }

    fftw_free(E);
    fftw_free(Eshift);

    fftw_destroy_plan(plan);
}


void ospr(cv::Mat& I, cv::Mat& H, bool Threading) {
    if (!Threading) {
        return ospr(I, H);
    }


    // Create Plan
    fftw_complex* Eshift = (fftw_complex*)fftw_malloc(sizeof(fftw_complex) * I.cols * I.rows);
    fftw_plan plan = fftw_plan_dft_2d(I.rows, I.cols, Eshift, Eshift, FFTW_BACKWARD, 0);
    fftw_free(Eshift);

    //random stuff
    std::mt19937_64 gen(123);
    std::uniform_real_distribution<double> dis(0.0, 2 * pi);


    std::thread threads[3*N];

    for (int i = 0; i < 3; i++)
    {
        // Fill with zeros

        for (int iteration = 0; iteration < N; iteration++)
        {

            /// We can split this into a thread
            // Spawn threads
            
            threads[i*N + iteration] = std::thread(&sub_ospr, std::ref(I), std::ref(H), i, iteration, std::ref(plan), std::ref(gen), std::ref(dis));
  
        }
    }

    for (std::size_t j = 0; j < 3 * N; j++) {
        threads[j].join();
    }
    

    fftw_destroy_plan(plan);
}


void sub_ospr(cv::Mat &I, cv::Mat &H, int i, int iteration, fftw_plan &plan, std::mt19937_64 &gen, std::uniform_real_distribution<double> &dis) {
    fftw_complex* E = (fftw_complex*)fftw_malloc(sizeof(fftw_complex) * I.cols * I.rows);
    fftw_complex* Eshift = (fftw_complex*)fftw_malloc(sizeof(fftw_complex) * I.cols * I.rows);
    

    uchar* p;
    double phi;
    std::complex<double> hpixel;


    for (std::size_t y = 0; y < I.rows; y++)
    {

        p = I.ptr<uchar>(y);
        for (std::size_t x = 0; x < I.cols; x++)
        {
            //Random Phase
            phi = dis(gen);
            hpixel = std::polar(sqrt(p[x]), phi);
            // Multiply by square root of image intensity
            E[y * I.cols + x][0] = hpixel.real();
            E[y * I.cols + x][1] = hpixel.imag();
        }
    }

    ifftshift(Eshift, E, I.cols, I.rows);
    fftw_execute_dft(plan, Eshift, Eshift);

    // This bit may cause problems - let's see!
    for (std::size_t y = 0; y < I.rows; y++)
    {
        p = H.ptr<uchar>(y);
        for (std::size_t x = 0; x < I.cols; x++)
        {
            if (Eshift[y * I.cols + x][1] > 0 || (Eshift[y * I.cols + x][1] == 0 && Eshift[y * I.cols + x][0] > 0))
            {
                p[3 * x + i] |= (1 << iteration);
            }
            else
            {
                p[3 * x + i] &= ~(1 << iteration);
            }
        }
    }

    fftw_free(E);
    fftw_free(Eshift);
}

void fillRandom(cv::Mat &I)
{
    uchar *p;
    std::mt19937 gen(123);
    std::uniform_int_distribution<int> dis(0, 255);

    for (std::size_t y = 0; y < I.rows; y++)
    {
        p = I.ptr<uchar>(y);
        for (std::size_t x = 0; x < I.cols; x++)
        {
            p[x] = dis(gen);
            //std::cout << (int)p[x] << '\n';
        }
    }
}

void fillConsecutive(cv::Mat& I)
{
    uchar* p;
    for (std::size_t y = 0; y < I.rows; y++)
    {
        p = I.ptr<uchar>(y);
        for (std::size_t x = 0; x < I.cols; x++)
        {
            p[x] = x + I.cols * y;
            //std::cout << (int)p[x] << '\n';
        }
    }
}

void fillConsecutive(fftw_complex* I, int width, int height) {
    for (std::size_t y = 0; y < height; y++) {
        for (std::size_t x = 0; x < width; x++) {
            I[y * width + x][0] = y * width + x;
            I[y * width + x][1] = 0;
        }
    }
}

void printComplex(const fftw_complex c) {
    std::cout << c[0] << "+" << c[1] << 'j';
}

void printMat(const fftw_complex* in, int width, int height) {
    for (std::size_t i = 0; i < height; i++) {
        std::cout << "[ ";
        for (std::size_t j = 0; j < width; j++) {
            printComplex(in[i * width + j]);
            std::cout << ", ";
        }
        std::cout << "\b\b]\n";
    }
    std::cout << "\n";
}

void printMat(cv::Mat &I) {
    uchar* p;
    for (std::size_t y = 0; y < I.rows; y++)
    {
        p = I.ptr<uchar>(y);
        std::cout << "[ ";
        for (std::size_t x = 0; x < I.cols; x++)
        {
            std::cout << (int)p[x];
            //std::cout << (int)p[x] << '\n';
            std::cout << ", ";
        }
        std::cout << "\b\b]\n";
    }
    std::cout << "\n";
}

// Implementation of fftshift and ifftshift

void ifftshift(fftw_complex* out, const fftw_complex* in, int width, int height)
{
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            out[width * ((y + height/2) % height) + ((x + width/2) % width)][0] = in[width * y + x][0];
            out[width * ((y + height / 2) % height) + ((x + width / 2) % width)][1] = in[width * y + x][1];
        }
    }
}
