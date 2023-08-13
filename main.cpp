#include <iostream>
#include <cassert>
#include <chrono>

#include <sys/types.h>
#include <signal.h>
#include <sys/wait.h>
#include <unistd.h>

#include "glpk.h"
#include "public.h"
#include "util.cpp"
#include "kruskal.cpp"

enum {
    CAP_SEC = 300,
};

bool part_belongs(const std::vector<int>& coloring, const std::vector<int>& edge, int c) {
    bool in = false, out = false;
    for (auto v : edge) {
        if (coloring[v] == c) {
            in = true;
        } else {
            out = true;
        }
    }
    return (in && out);
}

int color(const std::vector<std::vector<int>>& g, std::vector<int>& mark, int c, int v) {
    assert(!mark[v]);
    int ret = 1;
    mark[v] = c;
    for (const auto& u : g[v]) {
        if (!mark[u]) {
            ret += color(g, mark, c, u);
        }
    }
    return ret;
}

void rowgen(glp_tree *T, void *info) {
    auto lp = glp_ios_get_prob(T);
    auto& dataref = *static_cast<data *>(info);
    
    int n = dataref.n;
    int m = dataref.m;
    int t = dataref.t;
    auto& g = dataref.g;
    auto& conn = dataref.conn;

    for (int i = 0; i < n; ++i) {
        g[i].clear();
    }

    for (int j = 0; j < m; ++j) {
        if (glp_get_col_prim(lp, j + 1) > 0.01) {
            for (int i = 0; i < t - 1; ++i) {
                g[conn[j][i]].push_back(conn[j][i + 1]);
                g[conn[j][i + 1]].push_back(conn[j][i]);
            }
        }
    }

    std::vector<int> mark(n);

    int c = 0;
    for (int i = 0; i < n; ++i) {
        if (!mark[i]) {
            color(g, mark, ++c, i);
            std::vector<int> ind = { 0 };
            std::vector<double> coef = { 0.0 };
            for (int j = 0; j < m; ++j) {
                if (part_belongs(mark, conn[j], c)) {
                    ind.push_back(j + 1);
                    coef.push_back(1.0);
                }
            }
            if (ind.size() > 1) {
                int r = glp_add_rows(lp, 1);
                glp_set_row_bnds(lp, r, GLP_LO, 1.0, 1.0);
                glp_set_mat_row(lp, r, ind.size() - 1, &ind[0], &coef[0]);
            }
        }
    }
}

void heur(glp_tree *T, void *info) {
    auto lp = glp_ios_get_prob(T);
    auto dataptr = static_cast<data *>(info);
    
    int n = dataptr[0].n;
    int m = dataptr[0].m;
    int t = dataptr[0].t;
    auto& w = dataptr[0].w;
    auto& conn = dataptr[0].conn;

    auto& w_ = dataptr[1].w;
    auto& conn_ = dataptr[1].conn;
    auto& indices_ = dataptr[1].indices;

    DSU dsu(n);

    std::vector<double> ret(1 + m);

    double res = 0;

    for (int j = 0; j < m; ++j) {
        if (glp_get_col_prim(lp, j + 1) > 0.5) {
            for (int i = 0; i < t - 1; ++i) {
                dsu.connect(conn[j][i], conn[j][i + 1]);
            }
            ret[1 + j] = 1;
            res += w[j];
        }
    }

    for (int j = 0; j < m; ++j) {
        if (!belongs(dsu, conn_[j])) {
            for (int i = 0; i < t - 1; ++i) {
                dsu.connect(conn_[j][i], conn_[j][i + 1]);
            }
            ret[1 + indices_[j]] = 1;
            res += w_[j];
        }
    }

    glp_ios_heur_sol(T, &ret[0]);
}

void callback(glp_tree *T, void *info) {
    switch (glp_ios_reason(T)) {
    case GLP_IROWGEN:
        rowgen(T, info);
        break;
    case GLP_IBINGO:
        std::cout << "bingo\n";
        break;
    case GLP_IHEUR:
        heur(T, info);
        break;
    default:
        break;
    }
    return;
}

void calculate_and_output() {
    auto now = std::chrono::system_clock::now();

    auto data_ = new data[2];

    data& dr = data_[0];

    std::cin >> dr.n >> dr.m >> dr.t;
    dr.conn.resize(dr.m);
    dr.w.resize(dr.m);

    dr.g.resize(dr.n);
    for (auto& v : dr.g) {
        v.reserve(dr.n);
    }

    dr.indices.resize(dr.m);
    for (int i = 0; i < dr.m; ++i) {
        dr.indices[i] = i;
    }

    glp_prob *lp = glp_create_prob();
    glp_set_obj_dir(lp, GLP_MIN);
    glp_add_rows(lp, dr.n);
    for (int i = 0; i < dr.n; ++i) {
        glp_set_row_bnds(lp, 1 + i, GLP_LO, 1.0, 1.0);
    }

    glp_add_cols(lp, dr.m);

    for (int j = 0; j < dr.m; ++j) {
        glp_set_col_kind(lp, 1 + j, GLP_BV);
    }

    std::vector<int> ia(1 + dr.t * dr.m), ja(1 + dr.t * dr.m);
    std::vector<double> ar(1 + dr.t * dr.m);
    for (int j = 0; j < dr.m; ++j) {
        std::vector<int> v(dr.t);
        for (auto& x : v) {
            std::cin >> x;
            --x;
        }
        std::cin >> dr.w[j];
        dr.conn[j] = std::move(v);
        glp_set_obj_coef(lp, 1 + j, dr.w[j]);
        for (int i = 0; i < dr.t; ++i) {
            ia[1 + dr.t * j + i] = 1 + dr.conn[j][i];
            ja[1 + dr.t * j + i] = 1 + j;
            ar[1 + dr.t * j + i] = 1;
        }
    }

    glp_load_matrix(lp, dr.t * dr.m, &ia[0], &ja[0], &ar[0]);

    glp_iocp parm;
    glp_init_iocp(&parm);
    parm.cb_info = data_;
    parm.sr_heur = GLP_OFF;
    parm.br_tech = GLP_BR_DTH;
    parm.cb_func = callback;

    data_[1] = data_sort(data_[0]);

    glp_simplex(lp, GLP_PP_NONE);

    glp_intopt(lp, &parm);

    auto end = std::chrono::system_clock::now();

    dprintf(3, "%ld ", std::chrono::duration_cast<std::chrono::milliseconds>(end - now).count());

    dprintf(3, "%f\n", glp_mip_obj_val(lp));

    // std::vector<int> edges;

    // // for (int i = 0; i < dr.m; ++i) {
    // //     if (glp_mip_col_val(lp, i + 1)) {
    // //         dprintf(3, "%d ", i);
    // //         edges.push_back(i);
    // //     }
    // // }

    // DSU dsu(dr.n);

    // for (int j = 0; j < edges.size(); ++j) {
    //     for (int i = 0; i < dr.t - 1; ++i) {
    //         dsu.connect(dr.conn[edges[j]][i], dr.conn[edges[j]][i + 1]);
    //     }
    // }

    // // DSU dsu(dr.n);
    // // for (int j = 0; j < dr.m; ++j) {
    // //     if (glp_mip_col_val(lp, j + 1) == 1) {
    // //         for (int i = 0; i < dr.t - 1; ++i) {
    // //             dsu.connect(dr.conn[j][i], dr.conn[j][i + 1]);
    // //         }
    // //     }
    // // }

    // dprintf(3, "\n");

    // dprintf(3, "------------\n");

    // now = std::chrono::system_clock::now();

    // dsu = DSU(dr.n);
    // dprintf(3, "%f\n", kruskal(data_sort(dr), dsu));

    // end = std::chrono::system_clock::now();

    // dprintf(3, "Upper bound: elapsed %ld ms\n", std::chrono::duration_cast<std::chrono::milliseconds>(end - now).count());

    // dprintf(3, "------------\n");

    // now = std::chrono::system_clock::now();

    // dsu = DSU(dr.n);
    // dprintf(3, "%f\n", kruskal(data_sort(decompose(dr)), dsu) / (dr.t - 1));

    // end = std::chrono::system_clock::now();

    // dprintf(3, "Lower bound: elapsed %ld ms\n", std::chrono::duration_cast<std::chrono::milliseconds>(end - now).count());

    // // for (int i = 0; i < dr.n; ++i) {
    // //     dprintf(3, "%d %f; ", i, glp_mip_row_val(lp, i + 1));
    // // }
    // // dprintf(3, "\n");
    delete[] data_;
}

int main() {
    pid_t pid_a, pid_b;

    if (pid_a = fork()) {
        if (pid_b = fork()) {
            sleep(1);
            int status;
            waitpid(pid_a, &status, 0);

            if (WIFSIGNALED(status)) {
                dprintf(3, "%d\n", CAP_SEC * 1000);
            }
            
            kill(pid_a, SIGKILL);
            kill(pid_b, SIGKILL);
            waitpid(pid_b, &status, 0);
        } else {
            sleep(CAP_SEC);
            kill(pid_a, SIGKILL);
            return 0;
        }
    } else {
        calculate_and_output();
        return 0;
    }
}
