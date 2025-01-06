/*
Created on Jul 17 2024
@author: Nicola Cantarutti
*/

#include "BinaryPricer.hpp"

/***************************************************************
 * public method
 ***************************************************************/

double BinaryPricer::price(double spot, std::string method, std::string type, std::string expiry) {

    if (type != "call" && type != "put") {
        throw std::runtime_error("type must be call or put.");
    }
    if (expiry != "European" && expiry != "American") {
        throw std::runtime_error("expiry must be European or American.");
    }

    if (method == "closed") {
        if (expiry != "European") {
            throw std::runtime_error("expiry must be European when pricing with closed formula.");
        }
        else
            return closed_formula(spot, type);
    }

    else if (method == "MC") {
        if (expiry == "European")
            return MC_EU(spot, type);
        else
            return MC_AM(spot, type);
    }

    else if (method == "FDM") {
        return FDM(spot, type, expiry);
    }

    else if (method == "FEM") {
        return FEM(spot, type, expiry);
    }

    else
        throw std::runtime_error("method must be closed, FEM, FDM, MC.");
}

/***************************************************************
 * private methods
 ***************************************************************/

double BinaryPricer::closed_formula(double S0, std::string type) const {
    double d2 = (std::log(S0 / K) + (r - sig * sig / 2) * T) / (sig * std::sqrt(T));
    double N2 = 0.5 * (1 + std::erf(d2 / std::sqrt(2)));
    if (type == "call")
        return std::exp(-r * T) * N2;
    else
        return std::exp(-r * T) * (1 - N2);
}

double BinaryPricer::MC_EU(double S0, std::string type, int num_simulations,
                           unsigned int seed) const {

    double payoff_sum = 0.0;
    int num_threads = 0;

#pragma omp parallel
    {
#pragma omp atomic
        num_threads++;

        std::mt19937 gen(seed + omp_get_thread_num());
        std::normal_distribution<double> N(0.0, 1.0);
        double local_payoff_sum = 0.0;

#pragma omp for
        for (int i = 0; i < num_simulations; ++i) {
            double S_T = S0 * std::exp((r - 0.5 * sig * sig) * T + sig * std::sqrt(T) * N(gen));
            if (type == "call") {
                local_payoff_sum += (S_T > K) ? 1.0 : 0.0;
            }
            else {
                local_payoff_sum += (S_T <= K) ? 1.0 : 0.0;
            }
        }

#pragma omp atomic
        payoff_sum += local_payoff_sum;
    }

    double payoff_mean = payoff_sum / num_simulations;
    double price = std::exp(-r * T) * payoff_mean;
    // double std_err = std::exp(-r * T) * std::sqrt(payoff_mean * (1 - payoff_mean)) /
    // std::sqrt(num_simulations);

    // Store the number of threads used for testing purposes
    this->num_threads_used = num_threads; // member variables of the class

    return price;
}

double BinaryPricer::MC_AM(double S0, std::string type, int numPaths, int numSteps,
                           unsigned int seed) {

    std::vector<std::vector<double>> paths = generatePricePaths(S0, numPaths, numSteps, seed);
    std::vector<double> cashFlows(numPaths, 0.0);
    double dt = T / numSteps;

    for (int i = 0; i < numPaths; ++i) {
        if (type == "call") {
            cashFlows[i] = paths[i].back() > K ? 1.0 : 0.0;
        }
        else {
            cashFlows[i] = paths[i].back() <= K ? 1.0 : 0.0;
        }
    }

    for (int step = numSteps - 1; step > 0; --step) {
        std::vector<double> X, Y;
        for (int i = 0; i < numPaths; ++i) {
            bool inTheMoney = (type == "call") ? (paths[i][step] > K) : (paths[i][step] <= K);
            if (inTheMoney) {
                double discountedCashFlow = std::exp(-r * dt) * cashFlows[i];
                X.push_back(paths[i][step]);
                Y.push_back(discountedCashFlow);
            }
        }

        if (!X.empty()) {
            auto [beta0, beta1] = linearRegression(X, Y);
            for (int i = 0; i < numPaths; ++i) {
                bool inTheMoney = (type == "call") ? (paths[i][step] > K) : (paths[i][step] <= K);
                if (inTheMoney) {
                    double continuationValue = beta0 + beta1 * paths[i][step];
                    if (1.0 > continuationValue) { // 1.0 is the intrinsic value
                        cashFlows[i] = 1.0;        // optimal to excercise
                    }
                    else {
                        cashFlows[i] = std::exp(-r * dt) * cashFlows[i]; // optimal to wait
                    }
                }
                else {
                    cashFlows[i] = std::exp(-r * dt) * cashFlows[i];
                }
            }
        }
    }

    double oPrice = 0.0;
    for (double cashFlow : cashFlows) {
        oPrice += cashFlow;
    }
    oPrice /= numPaths;
    return oPrice * std::exp(-r * dt);
}

double BinaryPricer::FDM(double S0, std::string type, std::string expiry, const int Nspace,
                         const int Ntime) {
    double X0 = std::log(S0);

    double S_max = 3.0 * K;
    double S_min = K / 3.0;
    double x_max = std::log(S_max);
    double x_min = std::log(S_min);

    if (S0 <= S_min || S0 >= S_max) throw std::runtime_error("S0 not in the computation range");

    // space discretization
    std::vector<double> x(Nspace);
    double dx = (x_max - x_min) / (Nspace - 1);
    for (int j = 0; j < Nspace; ++j) {
        x[j] = x_min + j * dx;
    }

    // time discretization
    std::vector<double> T_array(Ntime);
    double dt = T / (Ntime - 1);
    for (int i = 0; i < Ntime; ++i) {
        T_array[i] = i * dt;
    }

    // Binary payoff
    std::vector<double> Payoff(Nspace);
    if (type == "call") {
        for (int j = 0; j < Nspace; ++j) {
            Payoff[j] = (std::exp(x[j]) > K) ? 1.0 : 0.0;
        }
    }
    else {
        for (int j = 0; j < Nspace; ++j) {
            Payoff[j] = (std::exp(x[j]) <= K) ? 1.0 : 0.0;
        }
    }

    // solution
    std::vector<double> V(Payoff);

    // tri-diagonal matrix constant coefficients
    double sig2 = sig * sig;
    double dxx = dx * dx;
    double a = (dt / 2) * ((r - 0.5 * sig2) / dx - sig2 / dxx);
    double b = 1 + dt * (sig2 / dxx + r);
    double c = -(dt / 2) * ((r - 0.5 * sig2) / dx + sig2 / dxx);

    double offset_min = (type == "put") ? a : 0;
    double offset_max = (type == "call") ? c : 0;

    // Backward iteration
    for (int i = Ntime - 2; i >= 0; --i) {
        V[0] -= offset_min;
        V[Nspace - 1] -= offset_max;

        if (expiry == "American")
            V = maximum(TDMA_simpler(a, b, c, V), Payoff);
        else
            V = TDMA_simpler(a, b, c, V);
    }

    // finds the option at S0 by interpolation
    double oPrice = 0.0;
    auto it = std::lower_bound(x.begin(), x.end(), X0);
    int ind = it - x.begin();
    if (X0 == x[ind])
        oPrice = V[ind];
    else
        oPrice = V[ind - 1] + (V[ind] - V[ind - 1]) / dx * (X0 - x[ind - 1]);

    return oPrice;
}

double BinaryPricer::FEM(double S0, std::string type, std::string expiry, const int Nspace,
                         const int Ntime) {

    int N; // number of finite elements
    if (Nspace % 2 == 0) {
        N = Nspace;
    }
    else {
        N = Nspace - 1;
    }

    const double a = 2 * r / (sig * sig);
    const double Tau = 0.5 * sig * sig * T;
    const double dt = Tau / Ntime; // time step size

    // Transformed lateral boundary
    auto BC = [a](double t, double x, double k) {
        return 1 / k * std::exp(0.5 * (a - 1) * x + 0.25 * (a + 1) * (a + 1) * t);
    };

    double x_max = std::log(3.0);
    double x_min = -x_max;
    double L = x_max - x_min;

    double X0 = std::log(S0 / K);
    if (X0 <= x_min || X0 >= x_max) throw std::runtime_error("S0 not in the computation range");

    auto x = generate_mesh(N, L);
    std::vector<double> h(N); // size of each element h
    for (int i = 0; i < N; ++i) {
        h[i] = x[i + 1] - x[i];
    }

    // Mass matrix
    std::vector<std::vector<double>> M_matrix(N + 1, std::vector<double>(N + 1, 0.0));
    // Stiffness matrix
    std::vector<std::vector<double>> K_matrix(N + 1, std::vector<double>(N + 1, 0.0));

    for (int i = 1; i < N; ++i) {
        double hi = h[i - 1];
        double hi1 = h[i];
        M_matrix[i][i] = (hi + hi1) / 3;
        M_matrix[i][i - 1] = hi / 6;
        M_matrix[i][i + 1] = hi1 / 6;
        K_matrix[i][i] = 1 * (1 / hi + 1 / hi1);
        K_matrix[i][i - 1] = -1 / hi;
        K_matrix[i][i + 1] = -1 / hi1;
    }

    M_matrix[0][0] = 1.0;
    M_matrix[N][N] = 1.0;
    K_matrix[0][0] = 1.0;
    K_matrix[N][N] = 1.0;

    // Payoff at time 0
    std::vector<double> Payoff(N + 1);
    if (type == "call") {
        for (int i = 0; i <= N; ++i) {
            if (x[i] < 0)
                Payoff[i] = 0;
            else
                Payoff[i] =
                    1 / K *
                    std::exp(0.5 * (a - 1) * x[i]); // Transformed initial boundary condition
        }
    }
    else {
        for (int i = 0; i <= N; ++i) {
            if (x[i] > 0)
                Payoff[i] = 0;
            else
                Payoff[i] = 1 / K * std::exp(0.5 * (a - 1) * x[i]);
        }
    }
    std::vector<double> u(Payoff);

    // Composition of matrix A
    std::vector<std::vector<double>> A(N + 1, std::vector<double>(N + 1));
    for (int i = 0; i <= N; ++i) {
        for (int j = 0; j <= N; ++j) {
            A[i][j] = M_matrix[i][j] + dt * K_matrix[i][j];
        }
    }
    // 3 diagonals of matrix A
    std::vector<double> aa(N), bb(N + 1), cc(N);
    for (int i = 0; i < N; ++i) {
        aa[i] = A[i + 1][i]; // subdiagonal
        bb[i] = A[i][i];     // diagonal
        cc[i] = A[i][i + 1]; // superdiagonal
    }
    bb[N] = A[N][N]; // last element of the diagonal

    double t = 0;
    while (t <= Tau) {

        if (type == "call") {
            u[0] = 0;
            u[N] = BC(t, x_max, K);
        }
        else {
            u[0] = BC(t, x_min, K);
            u[N] = 0;
        }

        std::vector<double> b(N + 1);
        for (int i = 1; i < N; ++i) {
            for (int j = i - 1; j <= i + 1; ++j) {
                b[i] += M_matrix[i][j] * u[j];
            }
        }

        if (expiry == "American") {
            for (int i = 0; i <= N; ++i) {
                Payoff[i] *=
                    std::exp(0.25 * (a + 1) * (a + 1) * dt); // Transformed Payoff is time dependent
            }
            u = maximum(TDMA(aa, bb, cc, b), Payoff);
        }
        else
            u = TDMA(aa, bb, cc, b);

        t += dt;
    }

    // finds the option at S0 by interpolation
    double oPrice = 0.0;
    auto it = std::lower_bound(x.begin(), x.end(), X0);
    int ind = it - x.begin();
    if (X0 == x[ind])
        oPrice = K * std::exp(-0.5 * (a - 1) * X0 - 0.25 * (a + 1) * (a + 1) * Tau) * u[ind];
    else {
        double u_mid = u[ind - 1] + (u[ind] - u[ind - 1]) / h[ind - 1] * (X0 - x[ind - 1]);
        oPrice = u_mid * K * std::exp(-0.5 * (a - 1) * X0 - 0.25 * (a + 1) * (a + 1) * Tau);
    }

    return oPrice;
}

/***************************************************************
 * Helper functions
 ***************************************************************/

std::vector<double> BinaryPricer::TDMA(const std::vector<double>& aa, const std::vector<double>& bb,
                                       const std::vector<double>& cc,
                                       const std::vector<double>& B) {
    int N = B.size();
    std::vector<double> d(B), x(N, 1.0), b(bb);

    // Overwrite coefficients
    for (int i = 1; i < N; ++i) {
        double w = aa[i - 1] / b[i - 1];
        b[i] -= w * cc[i - 1];
        d[i] -= w * d[i - 1];
    }

    // backward substitution
    x[N - 1] = d[N - 1] / b[N - 1];
    for (int i = N - 2; i >= 0; --i) {
        x[i] = (d[i] - cc[i] * x[i + 1]) / b[i];
    }

    return x;
}

std::vector<double> BinaryPricer::TDMA_simpler(double aa, double bb, double cc,
                                               const std::vector<double>& B) {
    int N = B.size();
    std::vector<double> b(N, bb);
    std::vector<double> d = B;
    std::vector<double> x(N, 1.0);

    // Overwrite coefficients
    for (int i = 1; i < N; ++i) {
        double w = aa / b[i - 1];
        b[i] = b[i] - w * cc;
        d[i] = d[i] - w * d[i - 1];
    }

    // backward substitution
    x[N - 1] = d[N - 1] / b[N - 1];
    for (int i = N - 2; i >= 0; --i) {
        x[i] = (d[i] - cc * x[i + 1]) / b[i];
    }

    return x;
}

std::vector<double> BinaryPricer::maximum(const std::vector<double>& a,
                                          const std::vector<double>& b) {
    std::vector<double> c(a.size());

    for (size_t i = 0; i < a.size(); ++i) {
        c[i] = std::max(a[i], b[i]);
    }

    return c;
}

std::vector<std::vector<double>> BinaryPricer::generatePricePaths(double S0, int numPaths,
                                                                  int numSteps, unsigned int seed) {
    std::vector<std::vector<double>> paths(numPaths, std::vector<double>(numSteps + 1));
    std::mt19937 generator(seed);
    std::normal_distribution<double> distribution(0.0, 1.0);

    double dt = T / numSteps;
    for (int i = 0; i < numPaths; ++i) {
        paths[i][0] = S0;
        for (int j = 1; j <= numSteps; ++j) {
            double dW = distribution(generator) * std::sqrt(dt);
            paths[i][j] = paths[i][j - 1] * std::exp((r - 0.5 * sig * sig) * dt + sig * dW);
        }
    }
    return paths;
}

std::pair<double, double> BinaryPricer::linearRegression(const std::vector<double>& X,
                                                         const std::vector<double>& Y) {
    int n = X.size();
    double meanX = std::accumulate(X.begin(), X.end(), 0.0) / n;
    double meanY = std::accumulate(Y.begin(), Y.end(), 0.0) / n;
    double S_XY = 0.0;
    double S_XX = 0.0;

    double diff_X; // temp variable
    for (int i = 0; i < n; ++i) {
        diff_X = (X[i] - meanX);
        S_XY += diff_X * (Y[i] - meanY);
        S_XX += diff_X * diff_X;
    }

    double beta = S_XY / S_XX;
    double alpha = (meanY - beta * meanX);

    return {alpha, beta};
}

std::vector<double> BinaryPricer::generate_mesh(int N, double L, double alpha) {

    if (alpha <= 0) throw std::runtime_error("alpha must be positive.");

    // Generate uniformly spaced nodes in [0, 1]
    int N2 = N / 2;
    std::vector<double> x(N2 + 1);
    for (int i = 0; i <= N2; ++i) {
        x[i] = static_cast<double>(i) / (N2);
    }

    // Apply non-uniform transformation and scale to the interval [0, L/2]
    for (int i = 0; i <= N2; ++i) {
        x[i] = 1 - std::tanh(alpha * (1 - x[i])) / std::tanh(alpha);
        x[i] = (L / 2) * x[i];
    }

    // Create full mesh
    std::vector<double> y(N + 1);
    std::copy(x.begin(), x.end(), y.begin() + N2);
    for (int i = 0; i < N2; ++i) {
        y[i] = -x[N2 - i];
    }

    return y;
}
