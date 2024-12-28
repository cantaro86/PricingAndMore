#include "Fibonacci.hpp"
#include <climits>
#include <cmath>
#include <complex>
#include <limits>
#include <numbers>

unsigned long fibonacci(unsigned long n) {

    unsigned long a = 0;
    unsigned long b = 1;
    unsigned long c = 0;

    for (unsigned long i = 0; i < n; ++i) {
        a = b;
        b = c;

        if (b > ULONG_MAX - a) {
            throw std::overflow_error("Overflow!");
        }

        c = a + b;
    }
    return c;
}

unsigned long fib_closed(unsigned long n) {

    using namespace std;
    using namespace numbers;
    using namespace literals::complex_literals;

    auto fib =
        2.0 / (sqrt(5) * pow(n, 1i)) * sinh(static_cast<double>(n) * log(1i * phi_v<double>));

    double result = std::round(std::abs(fib));

    if (result > static_cast<double>(numeric_limits<unsigned long>::max())) {
        throw std::overflow_error("Overflow!");
    }

    return static_cast<unsigned long>(result);
}
