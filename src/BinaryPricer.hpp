/*
Created on Jul 17 2024
@author: Nicola Cantarutti
*/

#ifndef BINARYPRICER_HPP
#define BINARYPRICER_HPP

#include <iostream>
#include <random>
#include <stdexcept>
#include <vector>

/**
 * @class BinaryPricer
 * @brief A class for pricing binary options.
 *
 * doxygen only here.
 */
class BinaryPricer {

  public:
    BinaryPricer(double strike, double maturity, double rate, double volatility)
        : K(strike), T(maturity), r(rate), sig(volatility) {}

    double price(double spot, std::string method, std::string type, std::string expiry);

  private:
    // Pricers with different numerical methods
    double closed_formula(double S0, std::string type) const;
    double MC_EU(double S0, std::string type, int num_simulations = 10000000,
                 unsigned int seed = 12345) const;
    double MC_AM(double S0, std::string type, int numPaths = 5000, int numSteps = 10000,
                 unsigned int seed = 54321);
    double FDM(double S0, std::string type, std::string expiry, const int Nspace = 6000,
               const int Ntime = 6000);
    double FEM(double S0, std::string type, std::string expiry, const int Nspace = 6000,
               const int Ntime = 6000);

    // Helper functions

    // Tridiagonal matrix algorithm
    std::vector<double> TDMA(const std::vector<double>& aa, const std::vector<double>& bb,
                             const std::vector<double>& cc, const std::vector<double>& B);
    // Tridiagonal matrix algorithm. Simplified for constant coefficients
    std::vector<double> TDMA_simpler(double aa, double bb, double cc, const std::vector<double>& B);
    // Elementwise maximum between 2 arrays
    std::vector<double> maximum(const std::vector<double>& a, const std::vector<double>& b);
    // Generate matrix containing GBM sample paths (each row is a path)
    std::vector<std::vector<double>> generatePricePaths(double S0, int paths, int steps,
                                                        unsigned int seed);
    // Returns linear regression parameters {alpha, beta}
    std::pair<double, double> linearRegression(const std::vector<double>& X,
                                               const std::vector<double>& Y);
    // Generate a non uniform and symmetric mesh with more nodes around the center.
    std::vector<double> generate_mesh(int N, double L, double alpha = 4);

    double K;   // strike price
    double T;   // maturity time
    double r;   // interest rate.
    double sig; // volatility
};

#endif

/* COMMENTS:

MC AM:
The Monte Carlo method for American option follows the Longstaff-Schwartz algorithm.
I implemented a linear regression with a single predictor, which gives good results.
But a production implementation should use at least 2 predictors.


FDM:
I am more familiar with FDM, so I decided to include also this method as an extra. It is very fast!
- I solved the Black-Scholes equation in log-variables. In this way the tri-diagonal matrix
coefficients are constant (in time and log-price).
- I used a fully implicit scheme, which is oscillation-free and unconditionally stable. Ideal for
discontinuaous payoff.
- Central difference approximation for first order space derivative.
- I decided to not use other linear algebra libraries, such as eigen.
- The Thomas algorithm (TDMA) is very fast, and easy to write. (same speed as LU usually)
- My implementation does not store the price values over time.  It is very memory efficient. Speed
can be improved.

FEM:
I am not very familiar with FEM.  Given the short time I had to find some tricks to solve the
problem in a quick way. Online I could only find quick information about FEM methods for heat
equation. I had no time to derive the discretization for Black-Scholes. For this reason I decided to
map the Black-Scholes equation to the Heat equation. (the change of variables can be found online or
in Wilmott 1994 ("Option pricing: Mathematical models and computation ") Section 5.4) The current
approach uses piecewise linear basis functions. I do not write the derivation here, but I have it on
my PC.

In the same book I found also that there are many approaches for computing American options using
FEM. But I used the simple "stopping time" approach used for FDM.

Advantages of the change of variables:
- the equation is dimensionless
- the domain is symmetric
- the discontinuity is at zero

I decided to have a node at zero.
I implemented a symmetric and non-uniform mesh, with more nodes near the discontinuity. The function
"generate_mesh" depends on a parameter alpha (alpha close to zero means almost uniform). There is a
small improvement with a bigger alpha near the discontinuity.

Mass and Stiffness matrices are sparse.  I didn't want to work with sparse matrices (Eigen?) so I
used normal dense matrices. This consumes some memory.  However the algorithm that I wrote is quite
efficient and considers only the diagonal terms. This can be improved.

In both FDM and FEM the choice of the domain [K/3, 3K] is hard-coded here. Here I didn't want to
have too many args in the function. In a production system it could be a good idea to create a
config.json file. It can contain the discretization parameters, the seed, mesh parameters, etc.

*/
