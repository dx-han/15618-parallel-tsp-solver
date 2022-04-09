#include <getopt.h>
#include <stdio.h>
#include <stdlib.h>
#include <string>
#include <iostream>
#include <fstream>
#include <vector>

void heldKarpCuda(int N, int *x, int *y, int *result, float *total_cost);
void printCudaInfo();

// return GB/s
float toBW(int bytes, float sec) {
    return static_cast<float>(bytes) / (1024. * 1024. * 1024.) / sec;
}

void usage(const char *progname) {
    printf("Usage: %s [options]\n", progname);
    printf("Program Options:\n");
    printf("  -f  --file             Input file\n");
    printf("  -?  --help             This message\n");
}

int main(int argc, char **argv) {
    std::string filename;
    int N, maxX, maxY;
    // Parse commandline options
    int opt;
    static struct option long_options[] = {
        {"file", 1, 0, 'f'},
        {"help", 0, 0, '?'},
        {0, 0, 0, 0}
    };

    while ((opt = getopt_long(argc, argv, "?f:", long_options, NULL)) != EOF) {
        switch (opt) {
        case 'f':
            filename = optarg;
            break;
        case '?':
        default:
            usage(argv[0]);
            return 1;
        }
    }
    // End parsing of commandline options
    
    std::fstream fs(filename);
    if (!fs) {
        std::cout << "Unknown file: " << filename << std::endl;
        return 1;
    }
    std::vector<int> x;
    std::vector<int> y;
    fs >> N >> maxX >> maxY;
    x.resize(N);
    y.resize(N);
    for (int i = 0; i < N; ++i) {
        fs >> x[i] >> y[i];
    }

    printCudaInfo();
    if (N <= 1) {
        std::cout << "N <= 1, early terminate." << std::endl;
        return 0; 
    }
    if (N > 32) {
        std::cout << "N is too large." << std::endl;
        return 1;
    }
    std::cout << "GPU memory required: " << (N - 1) * sizeof(float) * (1ULL << (N - 1)) / 1024. / 1024. << "MB" << std::endl;
    std::vector<int> result(N, 0);
    float total_cost;
    heldKarpCuda(N, x.data(), y.data(), result.data(), &total_cost);
    std::cout << "Path: ";
    for (int i = 0; i < N; ++i) {
        std::cout << result[i] << "->";
    }
    std::cout << 0 << std::endl << "Total cost: "<< total_cost << std::endl;

    std::ofstream ofs("output.txt");
    for (int i = 0; i < N; ++i) {
        ofs << result[i] << " ";
    }
    ofs << std::endl << total_cost << std::endl;

    return 0;
}
