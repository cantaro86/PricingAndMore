cmake_minimum_required(VERSION 3.22)

project(BinaryPricer)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# set(COMPILE_OPTIONS)
# if (CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
#     list(APPEND COMPILE_OPTIONS -Wall -Wextra -Werror -pedantic)
# elseif (CMAKE_CXX_COMPILER_ID MATCHES "MSVC")
#     list(APPEND COMPILE_OPTIONS /W4 /WX)
# endif()
# add_compile_options(${COMPILE_OPTIONS})


# set(LINK_OPTIONS)
# if (CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
#     list(APPEND LINK_OPTIONS -Wl,--no-as-needed)
# endif()

set(CMAKE_CONFIGURATION_TYPES Debug Release CI CACHE STRING "Choose the type of build." FORCE)
set(CMAKE_CXX_FLAGS_CI "-O2 -g -Wall -Wextra -Wpedantic -Werror -fsanitize=address,undefined")
set(CMAKE_C_FLAGS_CI "-O2 -g -Wall -Wextra -Wpedantic -Werror -fsanitize=address,undefined")


# The choice of the compiler and the build type must be done 
# externally (or by command line, VS code or config file)
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE CI CACHE STRING "Build type" FORCE)
endif()

message(STATUS "C++ compiler: ${CMAKE_CXX_COMPILER}")
message(STATUS "Compiler ID: " ${CMAKE_CXX_COMPILER_ID} )
message(STATUS "Build type: " ${CMAKE_BUILD_TYPE} )
message(STATUS "Compile options: ${COMPILE_OPTIONS}")
message(STATUS "Linker options: ${LINK_OPTIONS}")


add_executable(${PROJECT_NAME} src/main.cpp src/BinaryPricer.cpp)
# target_link_options(${PROJECT_NAME} PRIVATE ${LINK_OPTIONS}) 

add_executable(Fib src/main_fib.cpp src/Fibonacci.cpp)


# Add Catch2 as dependency
find_package(Catch2 QUIET)
if (NOT Catch2_FOUND)
    message(STATUS "Catch2 not found, fetching it...")
    include(FetchContent)
    set(FETCHCONTENT_BASE_DIR "${CMAKE_SOURCE_DIR}/.external")
    FetchContent_Declare(
        catch2
        GIT_REPOSITORY https://github.com/catchorg/Catch2.git
        GIT_TAG v3.8.0
    )
    FetchContent_MakeAvailable(catch2)
    message(STATUS "Catch2 install path: ${catch2_SOURCE_DIR}")
else()
    message(STATUS "Catch2 FOUND at ${Catch2_DIR}")
endif()

find_package(OpenMP REQUIRED)
if(OpenMP_CXX_FOUND)
    target_link_libraries(${PROJECT_NAME} OpenMP::OpenMP_CXX)
    # target_link_libraries(Fib OpenMP::OpenMP_CXX)
endif()

enable_testing()
include(CTest)

add_subdirectory(tests)
