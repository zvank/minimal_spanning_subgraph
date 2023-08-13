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
    // specifies optimise option: either 'b' for "bounded" or 'u' for "unbounded"
    char optimize_option;
    std::cin >> optimize_option;

    int size, limit;
    switch (optimize_option) {
        case 'b':
            std::cin >> size >> limit;
            break;
        case 'u':
            std::cin >> size;
            break;
        default:
            std::cerr << "Invalid optimize option\n";
            return 0;
    }

    std::vector<data_point> values(size);
    for (auto& [time, obj] : values) {
        std::cin >> time >> obj;
    }

    tdnp *ret;
    switch (optimize_option) {
        case 'b':
            ret = new tdnp(mle_optimiser_for_bounded_tdnp(values, limit));
            break;
        case 'u':
            ret = new tdnp(mle_optimiser_for_tdnp(values));
            break;
        default:
            return 0;
    }

    std::cout << ret->Ex << " "
        << ret->Ey << " "
        << ret->a  << " "
        << ret->b  << " "
        << ret->c  << "\n";

    delete ret;
}