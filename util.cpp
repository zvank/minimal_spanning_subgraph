#pragma once

#include "public.h"
#include <algorithm>
#include <iostream>

template<typename T>
std::ostream& operator<<(std::ostream& o, const std::vector<T>& v) {
    o << "{ ";
    for (const auto& x : v) {
        o << x << " ";
    }
    return o << "}";
}

data data_sort(const data& dr) {
    std::vector<std::pair<double, std::pair<std::vector<int>, int>>> zip(dr.m);
    for (int j = 0; j < dr.m; ++j) {
        zip[j] = { dr.w[j], { dr.conn[j], dr.indices[j] } };
    }
    std::sort(zip.begin(), zip.end());

    data ret;
    ret.n = dr.n;
    ret.m = dr.m;
    ret.t = dr.t;
    ret.conn.reserve(ret.m);
    ret.indices.reserve(ret.m);
    ret.w.reserve(ret.m);
    ret.g.resize(ret.n);
    for (auto& v : ret.g) {
        v.reserve(ret.n);
    }

    for (auto& [ w_, pair ] : zip) {
        auto& [ conn_, index_ ] = pair;
        ret.w.push_back(w_);
        ret.conn.push_back(conn_);
        ret.indices.push_back(index_);
    }

    return ret;
}

class DSU {
private:
    int n;
    std::vector<int> ptr;
    std::vector<int> w;

public:
    DSU(int n_) : n(n_), ptr(n_), w(n_, 1) {
        for (int i = 0; i < n_; ++i) {
            ptr[i] = i;
        }
    }

    void connect(int i, int j) {
        std::vector<int> to_be_merged;
        while (ptr[i] != i) {
            to_be_merged.push_back(i);
            i = ptr[i];
        }
        while (ptr[j] != j) {
            to_be_merged.push_back(j);
            j = ptr[j];
        }
        if (w[i] > w[j]) {
            for (auto& x : to_be_merged) {
                ptr[x] = i;
            }
            ptr[j] = i;
            w[i] += w[j];
        } else {
            for (auto& x : to_be_merged) {
                ptr[x] = j;
            }
            ptr[i] = j;
            w[j] += w[i];
        }
    }

    int comp(int i) {
        std::vector<int> to_be_merged;
        while (ptr[i] != i) {
            to_be_merged.push_back(i);
            i = ptr[i];
        }
        for (auto& x : to_be_merged) {
            ptr[x] = i;
        }
        return i;
    }

    // void relax() {
    //     for (int i = 0; i < n; ++i) {
    //         comp(i);
    //     }
    // }
};

bool belongs(DSU& dsu, const std::vector<int>& e) {
    int c = dsu.comp(e[0]);
    for (auto v : e) {
        if (c != dsu.comp(v)) {
            return false;
        }
    }
    return true;
}