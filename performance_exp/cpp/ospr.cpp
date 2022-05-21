#include <random>
#include <ctime>
//#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/imgcodecs.hpp>
#include <complex>
#include <fftw3.h>
#include <cstdlib>
#include <string>
#include <cmath>
#include <iostream>

#define HEIGHT 4
#define WIDTH 4

const int N = 8;
const int frames = 50;
const double pi = 3.141592654;


typedef fftw_complex matrix[HEIGHT][WIDTH];

void fillRandom(cv::Mat &image);
void ospr(cv::Mat &I, cv::Mat &H);
void ifftshift(fftw_complex *out, const fftw_complex *in, int xdim, int ydim);
void fillConsecutive(cv::Mat& I);
void printMat(const fftw_complex* in, int width, int height);
void fillConsecutive(fftw_complex* I, int width, int height);

int main(int argc, char const *argv[])
{
    // Write this to run like goldney's OSPR.py]
    /*
    std::cout << std::string(41, '#') << '\n'
        << "#\tOSPR Hologram Generator\t\t#\n"
        << std::string(41, '#') << '\n\n';

    */
    
    srand(time(NULL));
    


    cv::Mat image(HEIGHT, WIDTH, CV_8UC1);
    cv::Mat hologram(HEIGHT, WIDTH, CV_8UC3);

    //image = cv::imread("C:/Users/matth/Documents/IIB/Project/holography-project/images/transformed_grid.jpg", cv::IMREAD_GRAYSCALE);
    fillConsecutive(image);
    
    if (image.empty()) {//If the image is not loaded, show an error message//
        std::cout << "Couldn't load the image." << std::endl;
        system("pause");//pause the system and wait for users to press any key//
        return-1;
    }

    cv::imshow("Image1", image);

    ospr(image, hologram);

    if (hologram.empty()) {//If the image is not loaded, show an error message//
        std::cout << "OSPR Failed" << std::endl;
        system("pause");//pause the system and wait for users to press any key//
        return-1;
    }

    cv::imwrite("C:/Users/matth/Documents/IIB/Project/holography-project/holograms/holo.bmp", hologram);
    

    /*
    fftw_complex* in = (fftw_complex*)fftw_malloc(sizeof(fftw_complex) * WIDTH * HEIGHT);
    fftw_complex* out = (fftw_complex*)fftw_malloc(sizeof(fftw_complex) * WIDTH * HEIGHT);

    fillConsecutive(in, WIDTH, HEIGHT);
    ifftshift(out, in, WIDTH, HEIGHT);

    printMat(in, WIDTH, HEIGHT);
    printMat(out, WIDTH, HEIGHT);


    fftw_free(in);
    fftw_free(out);
    */

    return 0;
}

/*** Return a hologram of the same size as the image ***/
void ospr(cv::Mat &I, cv::Mat &H)
{
    fftw_complex* E = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * WIDTH * HEIGHT);
    fftw_complex* Eshift = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * WIDTH * HEIGHT);
    uchar *p;
    fftw_plan plan = fftw_plan_dft_2d(HEIGHT, WIDTH, Eshift, Eshift, FFTW_BACKWARD, 0);

    std::complex<double> hpixel;


    for (int i = 0; i < 3; i++)
    {
        // Fill with zeros

        for (int iteration = 0; iteration < N; iteration++)
        {
            for (std::size_t y = 0; y < HEIGHT; y++)
            {
                p = I.ptr<uchar>(y);
                for (std::size_t x = 0; x < WIDTH; x++)
                {
                    //Random Phase
                    double phi = ((double)rand() / (double)RAND_MAX) * 2 * pi;
                    std::complex<double> d = std::polar(1.0, phi);
                    // Multiply by square root of image intensity
                    hpixel = sqrt(p[x]) * d;
                    E[y * WIDTH + x][0] = hpixel.real();
                    E[y * WIDTH + x][1] = hpixel.imag();
                }
            }
            printMat(E, WIDTH, HEIGHT);
            // Next we perfofrm the inverse fourier transform of this
            ifftshift(Eshift, E, WIDTH, HEIGHT);

            printMat(Eshift, WIDTH, HEIGHT);

            fftw_execute(plan);

            printMat(Eshift, WIDTH, HEIGHT);

            // Maybe this is the point where it fails?
            for (std::size_t y = 0; y < HEIGHT; y++)
            {
                p = H.ptr<uchar>(y);
                for (std::size_t x = 0; x < WIDTH; x++)
                {
                    if (std::arg(*reinterpret_cast<std::complex<double> *>(&Eshift[y * HEIGHT + x])) > 0)
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

void fillRandom(cv::Mat &I)
{
    uchar *p;
    for (std::size_t y = 0; y < I.rows; y++)
    {
        p = I.ptr<uchar>(y);
        for (std::size_t x = 0; x < I.cols; x++)
        {
            p[x] = rand() % 256;
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
            p[x] = x + I.rows * y;
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
