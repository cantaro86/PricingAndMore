message(STATUS "Adding Tests")

add_executable(runFibTests test_Fib.cpp ../src/Fibonacci.cpp)
target_link_libraries(runFibTests PRIVATE Catch2::Catch2WithMain)
add_test(NAME Fibonacci COMMAND runFibTests)


add_executable(runClosedTests test_closed_formula.cpp ../src/BinaryPricer.cpp)
target_link_libraries(runClosedTests PRIVATE Catch2::Catch2WithMain OpenMP::OpenMP_CXX)
add_test(NAME Closed_formula COMMAND runClosedTests)


add_executable(runFDMTests test_FDM.cpp ../src/BinaryPricer.cpp)
target_link_libraries(runFDMTests PRIVATE Catch2::Catch2WithMain OpenMP::OpenMP_CXX)
add_test(NAME FDM COMMAND runFDMTests)


add_executable(runFEMTests test_FEM.cpp ../src/BinaryPricer.cpp)
target_link_libraries(runFEMTests PRIVATE Catch2::Catch2WithMain OpenMP::OpenMP_CXX)
add_test(NAME FEM COMMAND runFEMTests)


add_executable(runMCEUTests test_MC_EU.cpp ../src/BinaryPricer.cpp)
target_link_libraries(runMCEUTests PRIVATE Catch2::Catch2WithMain OpenMP::OpenMP_CXX)
add_test(NAME MC_EU COMMAND runMCEUTests)