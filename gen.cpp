#include <cmath>
#include <ctime>
#include <iostream>
#include <fstream>
#include <unordered_set>
#include <random>

constexpr uint64_t factorial(int n) {
    uint64_t res = 1;
    for (int i = 2; i <= n; ++i) {
        res *= i;
    }
    return res;
}

int main() {
    std::srand(std::time(0));
    int n, t, c;
    std::cin >> n >> t >> c;
    int m = n * std::log(n);

    std::cout << n << " " << m << " " << t << "\n";
    for (int j = 0; j < m; ++j) {
        std::unordered_set<int> vertices;
        while (vertices.size() < t) {
            vertices.insert(std::rand() % n + 1);
        }
        double w = (double)std::rand() / RAND_MAX;
        w = std::pow(c * factorial(t) * std::log(n) * w, 1.0L / (t - 1)) / n;
        for (auto x : vertices) {
            std::cout << x << " ";
        }
        std::cout << w << "\n";
    }
}