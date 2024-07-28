#include "../src/BinaryPricer.hpp"
#include <catch2/catch_all.hpp>
// #include <catch2/catch_test_macros.hpp>
// #include <catch2/matchers/catch_matchers_floating_point.hpp>

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
    
    double S0 = GENERATE(80.0, 90.0, 100.0, 110.0, 120.0);
    double oPrice_1 = pricer.price(S0, "closed", "call", "European");
    double oPrice_2 = pricer.price(S0, "FDM", "call", "European");
    REQUIRE_THAT(oPrice_1,  Catch::Matchers::WithinAbs(oPrice_2, 0.0001) );
}

TEST_CASE_METHOD(OptionFixture, "Test Put") {
    double S0 = GENERATE(80.0, 90.0, 100.0, 110.0, 120.0);
    double oPrice_1 = pricer.price(S0, "closed", "put", "European");
    double oPrice_2 = pricer.price(S0, "FDM", "put", "European");
    REQUIRE_THAT(oPrice_1,  Catch::Matchers::WithinAbs(oPrice_2, 0.0001) );
}

TEST_CASE_METHOD(OptionFixture, "throws exception for wrong input") {
    
    double S0 = 120.0;
    REQUIRE_THROWS_AS(pricer.price(S0, "open", "put", "European"), std::runtime_error);
    REQUIRE_THROWS_AS(pricer.price(S0, "closed", "set", "European"), std::runtime_error);
    REQUIRE_THROWS_AS(pricer.price(S0, "FDM", "put", "African"), std::runtime_error);
}