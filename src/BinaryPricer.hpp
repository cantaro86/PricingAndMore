/*
Created on Jul 17 2024
@author: Nicola Cantarutti
*/

#ifndef BINARYPRICER_HPP
#define BINARYPRICER_HPP

#include <iostream>
#include <omp.h>
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
    int getNumThreadsUsed() const { return num_threads_used; }

  private:
    // Pricers with different numerical methods
    double closed_formula(double S0, std::string type) const;
    double MC_EU(double S0, std::string type, int num_simulations = 50000000,
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
    mutable int num_threads_used = 0; // Number of threads used
};

#endif // BINARYPRICER_HPP
