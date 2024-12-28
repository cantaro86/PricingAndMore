/*
Created on Jul 17 2024
@author: Nicola Cantarutti
*/

#include "BinaryPricer.hpp"

int main() {
    double strike = 100.0;
    double maturity = 1.0;   // 1 year
    double rate = 0.1;       // 10% interest rate
    double volatility = 0.2; // 20% volatility

    BinaryPricer pricer(strike, maturity, rate, volatility);

    double spot = 100.0; // spot price

    try {
        std::cout << "European:" << std::endl;
        double optionPrice = pricer.price(spot, "closed", "call", "European");
        std::cout << "Closed call: " << optionPrice << std::endl;
        double optionPrice2 = pricer.price(spot, "closed", "put", "European");
        std::cout << "Closed put: " << optionPrice2 << std::endl;

        double optionPrice4 = pricer.price(spot, "MC", "call", "European");
        std::cout << "EU MC call: " << optionPrice4 << std::endl;
        double optionPrice5 = pricer.price(spot, "MC", "put", "European");
        std::cout << "EU MC put: " << optionPrice5 << std::endl;

        double optionPrice6 = pricer.price(spot, "FDM", "call", "European");
        std::cout << "EU FDM call: " << optionPrice6 << std::endl;
        double optionPrice7 = pricer.price(spot, "FDM", "put", "European");
        std::cout << "EU FDM put: " << optionPrice7 << std::endl;
        double optionPrice68 = pricer.price(spot, "FEM", "call", "European");
        std::cout << "EU FEM call: " << optionPrice68 << std::endl;
        double optionPrice67 = pricer.price(spot, "FEM", "put", "European");
        std::cout << "EU FEM put: " << optionPrice67 << std::endl;

        std::cout << "American:" << std::endl;
        double optionPrice8 = pricer.price(spot, "FDM", "call", "American");
        std::cout << "AM FDM call: " << optionPrice8 << std::endl;
        double optionPrice9 = pricer.price(spot, "FDM", "put", "American");
        std::cout << "AM FDM put: " << optionPrice9 << std::endl;
        double optionPrice88 = pricer.price(spot, "MC", "call", "American");
        std::cout << "AM MC call: " << optionPrice88 << std::endl;
        double optionPrice99 = pricer.price(spot, "MC", "put", "American");
        std::cout << "AM MC put: " << optionPrice99 << std::endl;
        double optionPrice69 = pricer.price(spot, "FEM", "call", "American");
        std::cout << "AM FEM call: " << optionPrice69 << std::endl;
        double optionPrice70 = pricer.price(spot, "FEM", "put", "American");
        std::cout << "AM FEM put: " << optionPrice70 << std::endl;
    }
    catch (const std::runtime_error& e) {
        std::cerr << "Error: " << e.what() << std::endl;
    }
    catch (const std::exception& e) {
        std::cerr << "Exception: " << e.what() << std::endl;
    }

    return 0;
}
