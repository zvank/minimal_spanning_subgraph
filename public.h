#pragma once

#include <vector>

struct data {
    int n, m, t;
    std::vector<std::vector<int>> conn;
    std::vector<double> w;
    std::vector<int> indices;
    std::vector<std::vector<int>> g;
};
