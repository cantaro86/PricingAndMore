#include "../src/BinaryPricer.hpp"
#include <catch2/catch_all.hpp>

class OptionFixture {
  public:
    double strike = 100.0;
    double maturity = 1.0;
    double rate = 0.1;
    double volatility = 0.2;
    BinaryPricer pricer;

    OptionFixture() : pricer(strike, maturity, rate, volatility) {}
};

TEST_CASE_METHOD(OptionFixture, "MC_EU Call Option Consistency") {
    double S0 = GENERATE(80.0, 90.0, 100.0, 110.0, 120.0);
    double oPrice1 = pricer.price(S0, "MC", "call", "European");
    double oPrice2 = pricer.price(S0, "closed", "call", "European");
    REQUIRE_THAT(oPrice1, Catch::Matchers::WithinAbs(oPrice2, 0.01));
}

TEST_CASE_METHOD(OptionFixture, "MC_EU Put Option Consistency") {
    double S0 = GENERATE(80.0, 90.0, 100.0, 110.0, 120.0);
    double oPrice1 = pricer.price(S0, "MC", "put", "European");
    double oPrice2 = pricer.price(S0, "closed", "put", "European");
    REQUIRE_THAT(oPrice1, Catch::Matchers::WithinAbs(oPrice2, 0.01));
}

TEST_CASE_METHOD(OptionFixture, "MC_EU Parallel Execution") {
    double S0 = 100.0;
    int num_threads = 0;

#pragma omp parallel
    {
#pragma omp atomic
        num_threads++;
    }

    REQUIRE(num_threads > 1); // Ensure that more than one thread is used

    double oPrice = pricer.price(S0, "MC", "call", "European");
    REQUIRE(oPrice > 0.0);

    int threads_used = pricer.getNumThreadsUsed();
    int max_threads = omp_get_max_threads();
    REQUIRE(threads_used ==
            max_threads); // Ensure that the number of threads used is equal to the maximum
}
