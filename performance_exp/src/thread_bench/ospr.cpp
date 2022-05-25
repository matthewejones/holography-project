#include "hol.h"
#include <vector>






void run(int height, int width, fftw_plan plan);



int main(int argc, char const* argv[])
{
    std::size_t height = 512;
    std::size_t width = 512;
    int frames = 50;

    if (argc == 1)
        return 1;
    else if (argc == 2) {
        width = atoi(argv[1]);
        height = width;
    }
    else if (argc >= 3) {
        width = atoi(argv[1]);
        height = atoi(argv[2]);
    }
    if (argc > 3)
        frames = atoi(argv[3]);


    fftw_complex* n = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * width * height);

    fftw_plan plan = fftw_plan_dft_2d(height, width, n, n, FFTW_BACKWARD, 0);

    // Start threads

    std::vector<std::thread> threads;

    for (int i = 0; i < frames; i++) {
        threads.push_back(std::thread(run, height, width, plan));
    }

    // Wait for threads to join

    for (int i = 0; i < frames; i++) {
        threads[i].join();
    }

    fftw_free(n);
    fftw_destroy_plan(plan);

    return 0;
}

void run(int height, int width, fftw_plan plan) {
    cv::Mat image(height, width, CV_8UC1);
    cv::Mat hologram(height, width, CV_8UC3);
    std::cout << "In run(): the pointers to image and hologram" << &image << &hologram << std::endl;

    fillRandom(image);
    ospr(image, hologram, plan);
}
