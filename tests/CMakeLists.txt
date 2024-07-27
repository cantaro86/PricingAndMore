message(STATUS "Adding Tests")

add_executable(runClosedTests test_closed_formula.cpp ../src/BinaryPricer.cpp)
target_link_libraries(runClosedTests PRIVATE Catch2::Catch2WithMain)
add_test(NAME Closed_formula COMMAND runClosedTests)



add_executable(runFibTests test_Fib.cpp ../src/Fibonacci.cpp)
target_link_libraries(runFibTests PRIVATE Catch2::Catch2WithMain)
add_test(NAME Fibonacci COMMAND runFibTests)



