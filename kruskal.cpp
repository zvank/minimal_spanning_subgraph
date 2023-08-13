#include "public.h"
#include "util.cpp"

data decompose(const data& dr) {
    data ret;
    ret.n = dr.n;
    ret.m = dr.m * (dr.t - 1);
    ret.t = 2;

    ret.w.reserve(ret.m);
    ret.conn.reserve(ret.m);
    ret.indices.reserve(ret.m);
    for (int i = 0; i < ret.m; ++i) {
        ret.indices.push_back(i);
    }

    ret.g.resize(ret.n);
    for (auto& v : ret.g) {
        v.reserve(ret.n);
    }

    for (int j = 0; j < dr.m; ++j) {
        for (int i = 0; i < dr.t - 1; ++i) {
            ret.conn.push_back({ dr.conn[j][i], dr.conn[j][i + 1] });
            ret.w.push_back(dr.w[j]);
        }
    }

    return ret;
}

double kruskal(const data& sorted, DSU& dsu) {

    double result = 0;

    for (int j = 0; j < sorted.m; ++j) {
        if (!belongs(dsu, sorted.conn[j])) {
            for (int i = 0; i < sorted.t - 1; ++i) {
                dsu.connect(sorted.conn[j][i], sorted.conn[j][i + 1]);
            }
            result += sorted.w[j];
        }
    }

    return result;
}
