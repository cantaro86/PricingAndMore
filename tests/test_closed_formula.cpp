// #define CATCH_CONFIG_MAIN
// #include <catch2/catch_all.hpp>

#include "../src/BinaryPricer.hpp"
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

class OptionFixture {
  public:
    double strike = 100.0;
    double maturity = 1.0;
    double rate = 0.1;
    double volatility = 0.2;
    BinaryPricer pricer;

    OptionFixture() : pricer(strike, maturity, rate, volatility) {}
};

TEST_CASE_METHOD(OptionFixture, "Test Call") {
    double oPrice = pricer.price(100.0, "closed", "call", "European");
    REQUIRE_THAT(oPrice, Catch::Matchers::WithinAbs(0.59305, 0.0001));
}

TEST_CASE_METHOD(OptionFixture, "Test Put") {
    double oPrice = pricer.price(100.0, "closed", "put", "European");
    REQUIRE_THAT(oPrice, Catch::Matchers::WithinAbs(0.311787, 0.0001));
}
