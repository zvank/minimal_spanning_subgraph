#include <cmath>
#include <functional>
#include <iostream>
#include <vector>

/* This structure represents the parameters of a 
 * two-dimensional normal distribution:
 * 
 * +-+   +-+-+       +--+
 * |x|   |a|0|       |Ex|
 * +-+ = +-+-+ * v + +--+,
 * |y|   |c|b|       |Ey|
 * +-+   +-+-+       +--+
 * 
 * where v is a standart N(0, I).
 */

struct data_point {
    double log_time;
    double obj;
};

class TwoDimensionalNormalParameters {
public:
    TwoDimensionalNormalParameters(double Ex_, double Ey_, double a_, double b_, double c_)
        : Ex(Ex_), Ey(Ey_), a(a_), b(b_), c(c_) {}

    double Ex, Ey, a, b, c;

    TwoDimensionalNormalParameters operator+(const TwoDimensionalNormalParameters& other) const {
        return TwoDimensionalNormalParameters (Ex + other.Ex,
                     Ey + other.Ey,
                     a  + other.a ,
                     b  + other.b ,
                     c  + other.c );
    }

    TwoDimensionalNormalParameters operator*(double m) const {
        return TwoDimensionalNormalParameters (Ex * m,
                     Ey * m,
                     a  * m,
                     b  * m,
                     c  * m);    
    }

    double operator*(const TwoDimensionalNormalParameters& other) const {
        return Ex * other.Ex+
               Ey * other.Ey+
               a  * other.a +
               b  * other.b +
               c  * other.c;
    }

    double norm() const {
        return *this * *this;
    }
};

std::ostream& operator<<(std::ostream& o, const TwoDimensionalNormalParameters& p) {
    return o << p.Ex << " " << p.Ey << " " << p.a << " " << p.b << " " << p.c;
}

using tdnp = TwoDimensionalNormalParameters;

template<typename T, typename F>
concept measurable_vec_over = requires(T l, T r) {
    { l * r } -> std::same_as<F>;
} && requires(T l, T r) {
    { l + r } -> std::same_as<T>;
} && requires(T l, F r) {
    { l * r } -> std::same_as<T>;
} && requires(T p) {
    { p.norm() } -> std::same_as<F>;
};

// constexpr
double Fi(double x) {
    // std::cout << "Fi(" << x << ") = " << 0.5 * (1 + std::erf(x / std::sqrt(2))) << "\n";
    return 0.5 * (1 + std::erf(x / std::sqrt(2)));
}

// accepts gragient fuction and optimises until reached threshold of epsilon norm of grad
template <typename param_t>
    requires (measurable_vec_over <param_t, double>)
param_t gradient_descent(
    std::function<param_t(const param_t &)>& grad_f,
    std::function<double(const param_t &)>& likelihood,
    param_t initial,
    double eps = 1e-6
) {
    uint counter = 0;
    std::vector<param_t> arg_seq = { initial };
    std::vector<param_t> grads = { grad_f(initial) };
    param_t curr = initial;
    initial.norm();

    while (grads.back().norm() > eps) {
        // std::cout << "norm = " << grads.back().norm() << "\n";
        ++counter;
        // std::cout << "cc=" << counter << ", ";
        // double length = grads.back().norm() > std::sqrt(eps) ? 
        //     1.0 / counter / std::sqrt(grads.back().norm()) :
        //     0.5;
        // double length = std::min(0.001, 1 / std::sqrt(grads.back().norm()));

        param_t curr = arg_seq.back();
        param_t grad_neg = grads.back() * -1.;

        double length = 1. / std::sqrt(grad_neg.norm());
        param_t new_grad = grad_f(curr + grad_neg * length);
        // while (grad_neg * new_grad < -0.001 * grad_neg.norm() || grad_neg * new_grad > 0) {
        //     // std::cout << "norm relation " << grad_neg * new_grad / grad_neg.norm() << "\n";
        //     if (grad_neg * new_grad < 0) {
        //         // std::cout << "l= " << length << " <0\n";
        //         length *= 1.02;
        //         // std::cout << "l= " << length << "\n";
        //     } else {
        //         // std::cout << "l= " << length << " >0\n";
        //         length *= 0.98;
        //         // std::cout << "l= " << length << "\n";
        //     }
        //     new_grad = grad_f(curr + grad_neg * length);
        // }

        // std::cout << "likelihood " << likelihood(arg_seq.back()) << "\n";
        // std::cout << "likelihood " << likelihood(curr + grad_neg * 0.00001) << "\n";

        // std::cout << "LIKELYHOOD CURR " << likelihood(curr) << "\n";
        // std::cout << "CURR " << curr << "\n";
        // std::cout << "GRAG NEG " << grad_neg << "\n";
        // std::cout << "REAL GRAD " << compute_grad(likelihood, curr, 1e-9) << "\n";
        while (likelihood(curr + grad_neg * length) > likelihood(curr) - 0.5 * length * grad_neg.norm()) {
            // std::cout << "length = " << length << "\n";
            // std::cout << "test point is " << curr + grad_neg * length << "\n";
            // std::cout << "f is " << likelihood(curr + grad_neg * length) << "\n";
            // std::cout << "(should be at most " << likelihood(curr) - 0.8 * length * grad_neg.norm() << ")\n";
            if (likelihood(curr + grad_neg * length) > likelihood(curr)) {
                // std::cout << "g\n";
            }
            length *= 0.6;
        }
        
        // std::cout << "length=" << length << "\n";

        arg_seq.push_back(arg_seq.back() + (grads.back() * -length));
        grads.push_back(grad_f(arg_seq.back()));

        // std::cout << "grad ";
        // std::cout << arg_seq.back().Ex << " ";
        // std::cout << arg_seq.back().Ey << " ";
        // std::cout << arg_seq.back().a << " ";
        // std::cout << arg_seq.back().b << " ";
        // std::cout << arg_seq.back().c << " -> ";
        // std::cout << grads.back().Ex << " ";
        // std::cout << grads.back().Ey << " ";
        // std::cout << grads.back().a << " ";
        // std::cout << grads.back().b << " ";
        // std::cout << grads.back().c << "\n";

        // param_t actual = compute_grad(likelihood, arg_seq.back());
        // std::cout << "actual = ";
        // std::cout << actual.Ex << " ";
        // std::cout << actual.Ey << " ";
        // std::cout << actual.a << " ";
        // std::cout << actual.b << " ";
        // std::cout << actual.c << "\n";

        // std::cout << "likelihood " << likelihood(arg_seq.back()) << "\n";
    }

    // std::cout << "cc=" << counter << "\n";
    return arg_seq.back();
}

template <typename T>
T compute_grad(
    std::function<double(const T &)>& f,
    T p,
    double eps = 1e-6
) {
    T dEx = { eps, 0, 0, 0, 0 };
    T dEy = { 0, eps, 0, 0, 0 };
    T da  = { 0, 0, eps, 0, 0 };
    T db  = { 0, 0, 0, eps, 0 };
    T dc  = { 0, 0, 0, 0, eps };

    double Ex = (f(p + dEx) - f(p)) / eps;
    double Ey = (f(p + dEy) - f(p)) / eps;
    double a  = (f(p + da ) - f(p)) / eps;
    double b  = (f(p + db ) - f(p)) / eps;
    double c  = (f(p + dc ) - f(p)) / eps;

    return { Ex, Ey, a, b ,c };
}

tdnp mle_optimiser_for_tdnp(const std::vector<data_point>& v) {

    if (v.empty()) {
        return { 0, 0, 0, 0, 0 };
    }

    double X_2 = 0, X_1 = 0, Y_2 = 0, Y_1 = 0, C = 0;

    for (const auto& x : v) {
        X_2 += x.log_time * x.log_time;
        X_1 += x.log_time;
        Y_2 += x.obj * x.obj;
        Y_1 += x.obj;
        C   += x.log_time * x.obj;
    }

    uint n = v.size();
    X_2 /= n;
    X_1 /= n;
    Y_2 /= n;
    Y_1 /= n;
    C   /= n;

    std::function<tdnp(const tdnp &)> grad = [X_2, X_1, Y_2, Y_1, C](const tdnp &p) {
        // std::cout << "grad ";
        double Ex = p.Ex;
        // std::cout << Ex << " ";
        double Ey = p.Ey;
        // std::cout << Ey << " ";
        double a = p.a;
        // std::cout << a << " ";
        double b = p.b;
        // std::cout << b << " ";
        double c = p.c;
        // std::cout << c << " -> ";

        double Ex_ = (X_1 - Ex) / a / a * (1 + c * c / b / b) - (Y_1 - Ey) * c / a / b / b;
        double Ey_ = (Y_1 - Ey - (X_1 - Ex) * c / a) / b / b;
        double a_  = (1 + c * c / b / b) / a / a / a * (X_2 - 2 * X_1 * Ex + Ex * Ex) - c / a / a / b / b *
            (C - X_1 * Ey - Ex * Y_1 + Ex * Ey) - 1 / a;
        double b_  = ((X_2 - 2 * X_1 * Ex + Ex * Ex) * c * c / a / a + (Y_2 - 2 * Y_1 * Ey + Ey * Ey) -
            2 * (C - X_1 * Ey - Ex * Y_1 + Ex * Ey) * c / a) / b / b / b - 1 / b;
        double c_  = -c * (X_2 - 2 * X_1 * Ex + Ex * Ex) / a / a / b / b + (C - X_1 * Ey - Ex * Y_1 + Ex * Ey) / a / b / b;

        // std::cout << -Ex_ << " ";
        // std::cout << -Ey_ << " ";
        // std::cout << -a_ << " ";
        // std::cout << -b_ << " ";
        // std::cout << -c_ << "\n";

        return tdnp( -Ex_, -Ey_, -a_, -b_, -c_ );
    };

    std::function<double(const tdnp &)> likelihood = [X_2, X_1, Y_2, Y_1, C](const tdnp &p) {
        double Ex = p.Ex;
        double Ey = p.Ey;
        double a = p.a;
        double b = p.b;
        double c = p.c;
        return -(-std::log(std::abs(a)) - std::log(std::abs(b)) 
               - (X_2 - 2 * X_1 * Ex + Ex * Ex) / 2 / a / a * (1 + c * c / b / b)
               - (Y_2 - 2 * Y_1 * Ey + Ey * Ey) / 2 / b / b
               + (C - X_1 * Ey - Ex * Y_1 + Ex * Ey) * c / a / b / b)
               ;
    };

    return gradient_descent<tdnp>(grad, likelihood, tdnp( X_1, Y_1, 1, 1, 0 ));
}



tdnp mle_optimiser_for_bounded_tdnp(const std::vector<data_point>& v, double t)
{
    if (v.empty()) {
        return { 0, 0, 0, 0, 0 };
    }

    double X_2 = 0, X_1 = 0, Y_2 = 0, Y_1 = 0, C = 0;
    uint n = 0;

    for (const auto& x : v) {
        if (x.obj != -1) {
            X_2 += x.log_time * x.log_time;
            X_1 += x.log_time;
            Y_2 += x.obj * x.obj;
            Y_1 += x.obj;
            C   += x.log_time * x.obj;
            ++n;
        }
    }

    uint N = v.size();
    X_2 /= n;
    X_1 /= n;
    Y_2 /= n;
    Y_1 /= n;
    C   /= n;

    double rel = (double)N / n - 1;

    // std::cout << rel << "\n";

    std::function<tdnp(const tdnp &)> grad = [X_2, X_1, Y_2, Y_1, C, rel, t](const tdnp &p) {
        // std::cout << "grad ";
        double Ex = p.Ex;
        // std::cout << Ex << " ";
        double Ey = p.Ey;
        // std::cout << Ey << " ";
        double a = p.a;
        // std::cout << a << " ";
        double b = p.b;
        // std::cout << b << " ";
        double c = p.c;
        // std::cout << c << " -> ";

        double Ex_ = (X_1 - Ex) / a / a * (1 + c * c / b / b) - (Y_1 - Ey) * c / a / b / b
            + rel * std::exp(-(Ex - t) * (Ex - t) / 2 / a / a) / std::abs(a) / Fi((Ex - t) / std::abs(a)) / std::sqrt(2 * M_PI)
            ;
        double Ey_ = (Y_1 - Ey - (X_1 - Ex) * c / a) / b / b;
        double a_  = (1 + c * c / b / b) / a / a / a * (X_2 - 2 * X_1 * Ex + Ex * Ex) - c / a / a / b / b *
            (C - X_1 * Ey - Ex * Y_1 + Ex * Ey) - 1 / a
            + rel * std::exp(-(Ex - t) * (Ex - t) / 2 / a / a) * (t - Ex) / a / std::abs(a) / Fi((Ex - t) / std::abs(a)) / std::sqrt(2 * M_PI)
            ;
        double b_  = ((X_2 - 2 * X_1 * Ex + Ex * Ex) * c * c / a / a + (Y_2 - 2 * Y_1 * Ey + Ey * Ey) -
            2 * (C - X_1 * Ey - Ex * Y_1 + Ex * Ey) * c / a) / b / b / b - 1 / b;
        double c_  = -c * (X_2 - 2 * X_1 * Ex + Ex * Ex) / a / a / b / b + (C - X_1 * Ey - Ex * Y_1 + Ex * Ey) / a / b / b;

        // std::cout << -Ex_ << " ";
        // std::cout << -Ey_ << " ";
        // std::cout << -a_ << " ";
        // std::cout << -b_ << " ";
        // std::cout << -c_ << "\n";

        return tdnp( -Ex_, -Ey_, -a_, -b_, -c_ );
    };

    std::function<double(const tdnp &)> likelihood = [X_2, X_1, Y_2, Y_1, C, rel, t](const tdnp &p) {
        // std::cout << "log =" << std::log(0) << "\n";
        double Ex = p.Ex;
        double Ey = p.Ey;
        double a = p.a;
        double b = p.b;
        double c = p.c;
        if (Fi((Ex - t) / std::abs(a)) == 0) {
            return (double)INFINITY;
        }
        return -(-std::log(std::abs(a)) - std::log(std::abs(b)) 
               - (X_2 - 2 * X_1 * Ex + Ex * Ex) / 2 / a / a * (1 + c * c / b / b)
               - (Y_2 - 2 * Y_1 * Ey + Ey * Ey) / 2 / b / b
               + (C - X_1 * Ey - Ex * Y_1 + Ex * Ey) * c / a / b / b)
               - rel * std::log(Fi((Ex - t) / std::abs(a)))
               ;
    };

    return gradient_descent<tdnp>(grad, likelihood, tdnp( t, Y_1, 1, 1, 0 ));
}

int main() {
    // {
    //     auto ret = mle_optimiser_for_tdnp({{8.04783,2.31585},{7.71155,2.38579},{8.90205,2.49978},{7.0859,1.58729},{5.98896,2.19904},{5.84932,2.12411},{7.42118,2.41838},{6.96224,2.45391},{4.61512,1.92344},{7.99531,2.11461},{8.72988,2.35946},{6.22059,2.2454},{7.64012,2.03908},{9.42497,2.78632},{6.83411,1.92432},{6.39859,2.10748},{7.00033,1.96945},{6.2653,2.3271},{7.19818,2.0992},{12.3521,3.0283},{9.93537,2.73347},{6.35957,2.6486},{8.04847,2.12523},{9.14963,2.65165},{4.77068,2.09235},{4.79579,2.25438},{5.9162,2.04738},{4.95583,1.93739},{8.39593,2.33578},{4.63473,2.05923},{7.03351,2.3641},{8.29978,2.42062},{8.47115,2.5098},{4.66344,1.68964},{9.82596,2.52422},{5.73657,2.30156},{5.3033,1.92381},{7.11233,2.0326},{6.14419,2.26015},{6.82546,2.16179},{7.62949,2.15227},{6.00635,1.87033},{3.55535,2.24083},{5.68358,2.52887},{8.29904,2.26582},{7.06561,2.11919},{6.81124,2.20171},{5.90808,1.79194},{8.4797,2.3405},{3.04452,1.92162},{7.87436,2.46546},{6.23637,2.48218},{9.19644,2.44784},{5.67332,2.17289},{7.74846,2.36793},{7.93916,2.42734},{9.03026,2.68144},{10.6071,2.71198},{7.76896,2.53481},{9.46614,2.50101},{4.15888,2.33306},{6.79794,2.50053},{7.10085,2.1267},{7.21744,2.44888},{7.68937,2.34416},{7.22839,2.45164},{7.80914,2.38137},{6.56667,1.92304},{6.57368,1.96861},{2.56495,1.83132},{8.15536,2.52227},{5.23111,2.24391},{5.5835,1.92871},{6.92363,2.33027},{5.35186,1.80441},{7.34407,2.47456},{5.41165,2.28306},{6.99668,2.46472},{6.66823,2.14565},{7.74587,2.28687},{6.02102,2.53754},{8.42266,2.24779},{4.17439,1.85257},{9.71908,2.71138},{6.6995,2.03327},{3.52636,1.75623},{5.80513,2.1144},{7.10168,2.22658},{8.32869,2.42581},{7.33824,2.26698},{8.85195,2.50377},{6.47235,2.25099},{6.48158,2.38977},{3.58352,1.81964},{10.8222,2.92895},{8.19919,2.65719},{4.44265,2.22584},{6.63595,2.33481},{9.98276,2.54305},{7.53636,2.41126}});
    //     std::cout << ret.Ex << " "
    //             << ret.Ey << " "
    //             << ret.a  << " "
    //             << ret.b  << " "
    //             << ret.c  << "\n";
    // }
    {
        auto ret = mle_optimiser_for_bounded_tdnp({{8.85367,-1},{8.62551,2.51931},{8.85367,-1},{8.85367,-1},{7.1624,1.96982},{8.85367,-1},{7.50219,2.05059},{7.32647,2.16636},{8.85367,-1},{8.85367,-1},{8.85367,-1},{8.66268,2.3334},{8.01036,2.18132},{8.85367,-1},{7.22257,2.1121},{7.96589,2.25437},{8.85367,-1},{6.2186,1.70022},{8.85367,-1},{0.693147,0.107111},{8.85367,-1},{8.66389,2.35239},{8.85367,-1},{8.18479,2.54712},{8.85367,-1},{6.16121,2.12656},{6.83626,2.0315},{8.85367,-1},{8.64065,2.15397},{8.85367,-1},{8.85367,-1},{7.46451,2.02305},{8.85367,-1},{7.36645,2.19995},{7.18766,2.25226},{8.00336,2.0914},{7.35244,2.18784},{7.6212,2.14392},{8.85367,-1},{0.693147,0.342674},{8.85367,-1},{7.63385,2.03912},{8.85367,-1},{8.54403,2.5148},{8.85367,-1},{8.85367,-1},{0.693147,0.311327},{8.22336,2.15898},{5.99146,2.31985},{7.29029,2.34801},{8.85367,-1},{8.71785,2.19085},{8.33375,2.26759},{8.85367,-1},{8.48508,2.2638},{8.57206,2.16398},{8.13388,2.16662},{8.85367,-1},{8.85367,-1},{8.85367,-1},{8.85367,-1},{8.85367,-1},{8.85367,-1},{8.85367,-1},{8.85367,-1},{0.693147,0.324748},{7.88118,2.11551},{8.85367,-1},{0.693147,0.160722},{5.67332,2.01833},{8.85367,-1},{7.54221,2.33592},{8.85367,-1},{7.18083,2.14545},{7.52994,2.2993},{8.85367,-1},{0.693147,0.289291},{5.69709,1.99445},{7.95262,2.43064},{8.85367,-1},{8.85367,-1},{8.85367,-1},{6.10702,2.12669},{8.04751,2.21542},{8.85367,-1},{8.85367,-1},{0,0.892309},{8.85367,-1},{8.63177,2.21921},{5.92158,2.04067},{8.37378,1.9234},{8.85367,-1},{8.25349,2.60033},{4.85203,2.04624},{6.14204,2.19616},{8.85367,-1},{8.42924,2.42596},{8.69885,2.13085},{7.16549,2.28122},{8.50856,2.54179}}, 8.85367);
        std::cout << ret.Ex << " "
                << ret.Ey << " "
                << ret.a  << " "
                << ret.b  << " "
                << ret.c  << "\n";
    }
    {
        auto ret = mle_optimiser_for_bounded_tdnp({{9.21034,-1},{8.62551,2.51931},{9.1397,1.96641},{9.21034,-1},{7.1624,1.96982},{9.21034,-1},{7.50219,2.05059},{7.32647,2.16636},{8.87333,2.42227},{8.9153,2.12326},{9.21034,-1},{8.66268,2.3334},{8.01036,2.18132},{9.21034,-1},{7.22257,2.1121},{7.96589,2.25437},{9.21034,-1},{6.2186,1.70022},{9.21034,-1},{0.693147,0.107111},{9.17751,2.38389},{8.66389,2.35239},{8.98857,2.26521},{8.18479,2.54712},{9.21034,-1},{6.16121,2.12656},{6.83626,2.0315},{9.21034,-1},{8.64065,2.15397},{9.21034,-1},{9.21034,-1},{7.46451,2.02305},{9.21034,-1},{7.36645,2.19995},{7.18766,2.25226},{8.00336,2.0914},{7.35244,2.18784},{7.6212,2.14392},{9.21034,-1},{0.693147,0.342674},{9.21034,-1},{7.63385,2.03912},{9.21034,-1},{8.54403,2.5148},{8.95905,2.26664},{9.21034,-1},{0.693147,0.311327},{8.22336,2.15898},{5.99146,2.31985},{7.29029,2.34801},{9.21034,-1},{8.71785,2.19085},{8.33375,2.26759},{9.21034,-1},{8.48508,2.2638},{8.57206,2.16398},{8.13388,2.16662},{9.21034,-1},{9.21034,-1},{9.21034,-1},{9.21034,-1},{9.21034,-1},{9.21034,-1},{9.21034,-1},{9.21034,-1},{0.693147,0.324748},{7.88118,2.11551},{9.21034,-1},{0.693147,0.160722},{5.67332,2.01833},{9.21034,-1},{7.54221,2.33592},{9.21034,-1},{7.18083,2.14545},{7.52994,2.2993},{9.21034,-1},{0.693147,0.289291},{5.69709,1.99445},{7.95262,2.43064},{9.21034,-1},{9.21034,-1},{9.16952,2.14743},{6.10702,2.12669},{8.04751,2.21542},{8.9482,2.38407},{9.21034,-1},{0,0.892309},{9.15282,2.41088},{8.63177,2.21921},{5.92158,2.04067},{8.37378,1.9234},{9.21034,-1},{8.25349,2.60033},{4.85203,2.04624},{6.14204,2.19616},{9.21034,-1},{8.42924,2.42596},{8.69885,2.13085},{7.16549,2.28122},{8.50856,2.54179}}, 9.21034);
        std::cout << ret.Ex << " "
                << ret.Ey << " "
                << ret.a  << " "
                << ret.b  << " "
                << ret.c  << "\n";
    }
    {
        auto ret = mle_optimiser_for_bounded_tdnp({{9.64569,2.51498},{8.62551,2.51931},{9.1397,1.96641},{12.0243,2.62195},{7.1624,1.96982},{9.93271,2.22264},{7.50219,2.05059},{7.32647,2.16636},{8.87333,2.42227},{8.9153,2.12326},{9.36126,2.54165},{8.66268,2.3334},{8.01036,2.18132},{11.5406,2.54468},{7.22257,2.1121},{7.96589,2.25437},{9.7695,2.26759},{6.2186,1.70022},{12.6115,-1},{5.39816,2.10711},{9.17751,2.38389},{8.66389,2.35239},{8.98857,2.26521},{8.18479,2.54712},{9.41042,2.51815},{6.16121,2.12656},{6.83626,2.0315},{9.93818,2.57297},{8.64065,2.15397},{9.47532,2.4266},{9.46785,2.39097},{7.46451,2.02305},{12.0422,2.62175},{7.36645,2.19995},{7.18766,2.25226},{8.00336,2.0914},{7.35244,2.18784},{7.6212,2.14392},{12.6115,-1},{8.79966,2.34267},{9.56079,2.45072},{7.63385,2.03912},{9.5828,2.50294},{8.54403,2.5148},{8.95905,2.26664},{12.6115,-1},{9.43978,2.31133},{8.22336,2.15898},{5.99146,2.31985},{7.29029,2.34801},{10.4971,2.49684},{8.71785,2.19085},{8.33375,2.26759},{9.23873,2.26662},{8.48508,2.2638},{8.57206,2.16398},{8.13388,2.16662},{10.923,2.35369},{11.0809,2.58407},{9.23805,2.37665},{9.37509,1.83404},{9.40681,2.27513},{9.46715,2.16071},{10.4127,2.52463},{12.6115,-1},{8.90626,2.32475},{7.88118,2.11551},{12.6115,-1},{8.751,2.16072},{5.67332,2.01833},{10.1465,1.96207},{7.54221,2.33592},{9.33353,2.46917},{7.18083,2.14545},{7.52994,2.2993},{12.6115,-1},{10.5128,2.28929},{5.69709,1.99445},{7.95262,2.43064},{10.5986,2.47225},{9.72149,2.641},{9.16952,2.14743},{6.10702,2.12669},{8.04751,2.21542},{8.9482,2.38407},{12.6115,-1},{4.8828,1.89231},{9.15282,2.41088},{8.63177,2.21921},{5.92158,2.04067},{8.37378,1.9234},{12.081,2.30554},{8.25349,2.60033},{4.85203,2.04624},{6.14204,2.19616},{10.5174,2.3652},{8.42924,2.42596},{8.69885,2.13085},{7.16549,2.28122},{8.50856,2.54179}}, 12.6116);
        std::cout << ret.Ex << " "
                << ret.Ey << " "
                << ret.a  << " "
                << ret.b  << " "
                << ret.c  << "\n";
    }
}