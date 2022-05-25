#include "hol.h"

int main(int argc, char const* argv[])
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

    image = cv::imread("C:/Users/matth/Documents/IIB/Project/holography-project/images/transformed_grid.jpg", cv::IMREAD_GRAYSCALE);
    //fillConsecutive(image);

    if (image.empty()) {//If the image is not loaded, show an error message//
        std::cout << "Couldn't load the image." << std::endl;
        system("pause");//pause the system and wait for users to press any key//
        return-1;
    }

    //printMat(image);
    //cv::imshow("Image1", image);

    ospr(image, hologram);

    if (hologram.empty()) {//If the image is not loaded, show an error message//
        std::cout << "OSPR Failed" << std::endl;
        system("pause");//pause the system and wait for users to press any key//
        return-1;
    }
    //printMat(hologram);
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
