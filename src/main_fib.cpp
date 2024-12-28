#include "../src/Fibonacci.hpp"
#include <iostream>

int main() {

    std::cout << "Insert a positive number:" << std::endl;
    unsigned long n;
    std::cin >> n;

    try {
        std::cout << "the " << n << " number of Fibonacci sequence is: " << fibonacci(n)
                  << std::endl;
    }
    catch (const std::overflow_error& e) {
        std::cerr << "Error: " << e.what() << std::endl;
    }
    catch (const std::exception& e) {
        std::cerr << "Exception: " << e.what() << std::endl;
    }

    try {
        std::cout << "the " << n << " number of Fibonacci sequence is: " << fib_closed(n)
                  << std::endl;
    }
    catch (const std::overflow_error& e) {
        std::cerr << "Error: " << e.what() << std::endl;
    }
    catch (...) {
        std::cerr << "Unknown Exception" << std::endl;
    }

    return 0;
}