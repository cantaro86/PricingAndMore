#include "../src/Fibonacci.hpp"
#include <catch2/catch_test_macros.hpp>


TEST_CASE("Test closed formula for Fibonacci") {

    REQUIRE(fib_closed(1) == fibonacci(1));
    REQUIRE(fib_closed(2) == fibonacci(2));
    REQUIRE(fib_closed(3) == fibonacci(3));
    REQUIRE(fib_closed(4) == fibonacci(4));
    REQUIRE(fib_closed(5) == fibonacci(5));
    REQUIRE(fib_closed(6) == fibonacci(6));
    REQUIRE(fib_closed(7) == fibonacci(7));
    REQUIRE(fib_closed(8) == fibonacci(8));
    REQUIRE(fib_closed(9) == fibonacci(9));
    REQUIRE(fib_closed(10) == fibonacci(10));
    REQUIRE(fib_closed(40) == fibonacci(40));
    REQUIRE(fib_closed(60) == fibonacci(60));
}

TEST_CASE("fib_close throws exception for big number") {
    
    REQUIRE_THROWS_AS(fib_closed(100), std::runtime_error);
}

TEST_CASE("fibonacci throws exception for big number") {
    
    REQUIRE_THROWS_AS(fibonacci(100), std::runtime_error);
}