#include "image.h"
#include "util.cpp"

#include <vector>
#include <fstream>
#include <iostream>
#include <algorithm>
#include <cmath>

inline Image draw(const std::vector<uint>& sorted) {
    Image img(sorted.size() + 10, sorted.size() + 10);
    for (int i = 0; i < sorted.size(); ++i) {
        uint64_t y = sorted[i] * sorted.size() / sorted.back();
        img.SetPixel({255, 255, 255}, sorted.size() + 5 - y, 5 + i);
    }
    return img;
}

inline Image draw(const std::vector<double>& sorted, size_t argmin = 0, size_t argmax = -1) {
    argmax = (argmax == -1) ? sorted.size() - 1 : argmax;
    Image img(sorted.size() + 10, sorted.size() + 10);
    for (int i = 0; i < sorted.size(); ++i) {
        // std::cout << "a " << sorted[i] << " " << sorted[argmin] << " " << sorted[argmax] << "\n";
        int y = (sorted[i]-sorted[argmin]) * sorted.size() / (sorted[argmax]-sorted[argmin]);
        // std::cout << "b " << y << "\n";
        img.SetPixel({255, 255, 255}, sorted.size() + 5 - y, 5 + i);
        // std::cout << "c\n";
    }
    return img;
}

int main(int argc, char **argv) {
    std::ifstream in;
    in.open(argv[1]);

    int size, limit;
    in >> size >> limit;
    std::vector<uint> time(size);
    std::vector<double> obj(size);
    uint64_t total = 0;
    for (int i = 0; i < size; ++i) {
        in >> time[i];
        total += time[i];
        std::cout << "time[" << i << "] = " << time[i] << "\n";
        if (time[i] != limit) {
            in >> obj[i];
        }
    }
    std::sort(time.begin(), time.end());
    std::cout << "total = " << total << "\n";
    std::cout << "median = " << time[size / 2] << "\n";
    std::cout << "avg = " << (double)total / size << "\n";

    Image img = draw(time);

    img.Write("img.png");

    std::vector<double> log(size);
    for (int i = 0; i < size; ++i) {
        log[i] = std::log(time[i]);
    }
    
    Image img_ = draw(log);
    img_.Write("img_.png");

    std::vector<double> avg(300);
    std::vector<int> count(300);
    std::vector<double> sum(300);
    int current = 0;

    for (auto& x : time) {
        if (x / 1000 != current) {
            for (int i = current + 1; i <= x / 1000; ++i) {
                count[i] = count[current];
                sum[i] = sum[current];
            }
            current = x / 1000;
            if (x / 1000 >= 300) {
                break;
            }
        }
        ++count[current];
        sum[current] += x;
    }

    int argmin = 0;
    double min = 1000000000000.;
    int argmax = 0;
    double max = 0.;

    for (int i = 0; i < 300; ++i) {
        avg[i] = (count[i] == 0) ? 3000000 : (sum[i] / count[i] + 1000. * (i + 1) * (size - count[i]) / count[i]);
        if (avg[i] < min) {
            argmin = i;
            min = avg[i];
        }
        if (avg[i] > max) {
            argmax = i;
            max = avg[i];
        }
    }

    std::cout << argmin << " " << min << "\n";
    std::cout << argmax << " " << max << "\n";

    Image img__ = draw(avg, argmin, argmax);
    img__.Write("img__.png");

    // std::vector<double> avg(2000);
    // std::vector<int> count(2000);
    // std::vector<double> sum(2000);
    // int current = 0;

    // for (auto& x : time) {
    //     if (x != current) {
    //         for (int i = current + 1; i <= x && i < 2000; ++i) {
    //             count[i] = count[current];
    //             sum[i] = sum[current];
    //         }
    //         current = x;
    //         if (x >= 2000) {
    //             break;
    //         }
    //     }
    //     ++count[current];
    //     sum[current] += x;
    // }

    // std::cout << count << "\n" << sum << "\n";

    // int argmin = 0;
    // double min = 2000000000000.;
    // int argmax = 0;
    // double max = 0.;

    // for (int i = 0; i < 2000; ++i) {
    //     avg[i] = (count[i] == 0) ? 5000 : (sum[i] / count[i] + (i + 1) * (size - count[i]) / count[i]);
    //     if (avg[i] < min) {
    //         argmin = i;
    //         min = avg[i];
    //     }
    //     if (avg[i] > max) {
    //         argmax = i;
    //         max = avg[i];
    //     }
    // }

    // std::cout << avg << "\n";
    // std::cout << argmin << " " << min << "\n";
    // std::cout << argmax << " " << max << "\n";

    // Image img__ = draw(avg, argmin, argmax);
    // img__.Write("img__.png");
}