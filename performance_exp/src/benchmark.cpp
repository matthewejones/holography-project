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

#define HEIGHT 192
#define WIDTH 192

#define fftshift(out, in, x, y) circshift(out, in, x, y, (x / 2), (y / 2))
#define ifftshift(out, in, x, y) circshift(out, in, x, y, ((x + 1) / 2), ((y + 1) / 2))

const int N = 8;
const int frames = 50;
const double pi = 3.141592654;

typedef fftw_complex matrix[HEIGHT][WIDTH];

void fillRandom(cv::Mat &image);
void ospr(cv::Mat &I, cv::Mat &H);
template <typename ty>
void circshift(ty *out, const ty *in, int xdim, int ydim, int xshift, int yshift);

int main(int argc, char const *argv[])
{
    // Write this to run like goldney's OSPR.py]
    /*
    std::cout << std::string(41, '#') << '\n'
        << "#\tOSPR Hologram Generator\t\t#\n"
        << std::string(41, '#') << '\n\n';


    srand(time(NULL));
    */
    cv::Mat image(HEIGHT, WIDTH, CV_8UC1);
    cv::Mat hologram(HEIGHT, WIDTH, CV_8UC3);
    /*
    image = cv::imread("C:/Users/matth/Documents/IIB/Project/holography-project/images/grid.jpg", cv::IMREAD_GRAYSCALE);

    if (image.empty()) {//If the image is not loaded, show an error message//
        std::cout << "Couldn't load the image." << std::endl;
        system("pause");//pause the system and wait for users to press any key//
        return-1;
    }
    */
    //cv::imshow("Image", image);
    //cv::waitKey(0);

    
    fillRandom(image);

    for (int i = 0; i < frames; i++)
    {
        fillRandom(image);
        ospr(hologram, image);
    }

    

    ospr(image, hologram);

    /*
    if (hologram.empty()) {//If the image is not loaded, show an error message//
        std::cout << "OSPR Failed" << std::endl;
        system("pause");//pause the system and wait for users to press any key//
        return-1;
    }


    cv::imwrite("C:/Users/matth/Documents/IIB/Project/holography-project/images/holo.bmp", hologram);
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

    std::cout << "Image pixel: " << (int) I.at<uchar>(3, 5) << ", " << (int)I.at<uchar>(103, 532) << std::endl;

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
                    // There's a lot going on in this next line
                    // we are essentially timesing the sqrt of the image pixel and adding random phase
                    // rand() / rand_max will give a random number betwen 0 and 1
                    // pointer to a
                    //std::cout << "Current Pixel Value:" << (unsigned int)p[x] << std::endl;
                    double phi = ((double)rand() / (double)RAND_MAX) * 2 * pi;
                    std::complex<double> d = std::polar(1.0, phi);
                    //std::cout << phi << '\n';
                    // Problem here?
                    hpixel = sqrt(p[x]) * d;
                    // Test this line
                    //if (p[x] != 0)
                    //std::cout << "hpixel value: " << hpixel << std::endl;
                    // In binary the types are equivalent so we can just recast pointers to them
                    E[y * WIDTH + x][0] = hpixel.real();
                    E[y * WIDTH + x][1] = hpixel.imag();
                    //std::cout << "E pixel value: " << E[y * WIDTH + x][0] << " + " << E[y * WIDTH + x][1] << 'j' << std::endl;
                }
            }
            // Next we perfofrm the inverse fourier transform of this
            ifftshift(Eshift, E, WIDTH, HEIGHT);

            fftw_execute(plan);

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
            //
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

void getDiffusor(cv::Mat &I)
{
    uchar *p;
    for (std::size_t y = 0; y < I.rows; y++)
    {
        p = I.ptr<uchar>(y);
        for (std::size_t x = 0; x < I.cols; x++)
        {
            p[x] = rand() % 256;
        }
    }
}

void sqrt(cv::Mat &I)
{
    uchar *p;
    for (std::size_t y = 0; y < I.rows; y++)
    {
        p = I.ptr<uchar>(y);
        for (std::size_t x = 0; x < I.cols; x++)
        {
            p[x] = (uchar)sqrt(p[x]);
        }
    }
}

// Implementation of fftshift and ifftshift
template<class ty>
void circshift(ty* out, const ty* in, int xdim, int ydim, int xshift, int yshift)
{
    for (int i = 0; i < xdim; i++) {
        int ii = (i + xshift) % xdim;
        for (int j = 0; j < ydim; j++) {
            int jj = (j + yshift) % ydim;
            out[ii * ydim + jj][0] = in[i * ydim + j][0];
            out[ii * ydim + jj][1] = in[i * ydim + j][1];
        }
    }
}
