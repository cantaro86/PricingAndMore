Just a repo for my testing of C++

It includes 

- Binary option pricer for European and American options. 
  It uses Monte Carlo, Finite differences methods and Finite Elements methods.

- The discretization for FEM is described in docs. For the FEM approach, I first map the problem to 
  a heat equation and then use the FEM discretization.

- A closed formula (non-recursive) for the Fibonacci numbers.


SET:

Better to use:

```bash
export CXX=$(which clang++)
export CC=$(which clang)
```

for portability.  Otherwise use:

```cmake 
find_program(CLANGXX clang++)
if(CLANGXX)
    set(CMAKE_CXX_COMPILER ${CLANGXX})
else()
    message(FATAL_ERROR "clang++ not found")
endif() 
```

or 

```cmake
execute_process(COMMAND bash -c "which clang++" OUTPUT_VARIABLE CXX_COMPILER_PATH OUTPUT_STRIP_TRAILING_WHITESPACE)
set(CMAKE_CXX_COMPILER ${CXX_COMPILER_PATH})
```


COMMENTS:

MC AM:
The Monte Carlo method for American option follows the Longstaff-Schwartz algorithm.
I implemented a linear regression with a single predictor, which gives good results.
But a production implementation should use at least 2 predictors.


FDM:
I am more familiar with FDM, so I decided to include also this method as an extra. It is very fast!
- I solved the Black-Scholes equation in log-variables. In this way the tri-diagonal matrix
coefficients are constant (in time and log-price).
- I used a fully implicit scheme, which is oscillation-free and unconditionally stable. Ideal for
discontinuaous payoff.
- Central difference approximation for first order space derivative.
- I decided to not use other linear algebra libraries, such as eigen.
- The Thomas algorithm (TDMA) is very fast, and easy to write. (same speed as LU usually)
- My implementation does not store the price values over time.  It is very memory efficient. Speed
can be improved.

FEM:
I am not very familiar with FEM.  
I decided to
map the Black-Scholes equation to the Heat equation. (the change of variables can be found online or
in Wilmott 1994 ("Option pricing: Mathematical models and computation ") Section 5.4)   
The current
approach uses piecewise linear basis functions. See docs.

In the same book I found also that there are many approaches for computing American options using
FEM. But I used the simple "stopping time" approach used for FDM.

Advantages of the change of variables:
- the equation is dimensionless
- the domain is symmetric
- the discontinuity is at zero

I decided to have a node at zero.
I implemented a symmetric and non-uniform mesh, with more nodes near the discontinuity. The function
"generate_mesh" depends on a parameter alpha (alpha close to zero means almost uniform). There is a
small improvement with a bigger alpha near the discontinuity.

Mass and Stiffness matrices are sparse.  I didn't want to work with sparse matrices (Eigen?) so I
used normal dense matrices. This consumes some memory.  However the algorithm that I wrote is quite
efficient and considers only the diagonal terms. This can be improved.

In both FDM and FEM the choice of the domain [K/3, 3K] is hard-coded here. Here I didn't want to
have too many args in the function. In a production system it could be a good idea to create a
config.json file. It can contain the discretization parameters, the seed, mesh parameters, etc.
