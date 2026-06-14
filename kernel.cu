#include "state_solver.h"
#include "xoro_matrix.h"
#include <stdlib.h>
#include <stdio.h>

/*
We're targetting the following sequence of nextInt calls:

1.  nextInt(50) <= 32
(skip 1 call between consecutive target nextInt calls)
2.  nextInt(50) <= 16
3.  nextInt(50) <= 8
4.  nextInt(50) <= 4
5.  nextInt(50) <= 2
6.  nextInt(50) <= 1
7.  nextInt(50) == 0
8.  nextInt(50) == 0
...
12. nextInt(50) == 0 <-- all
13. nextInt(50) == 0 <-- of these
14. nextInt(50) == 0 <-- calls are
15. nextInt(50) == 0 <-- partially
16. nextInt(50) == 0 <-- reversed
17. nextInt(50) == 0 <--
18. nextInt(50) == 0 <--
*/

/*
We're setting the initial xoroshiro state to be at call #7.
Firstly, we construct a system of GF(2) equations that gives a partial reversal
of calls 12-18. This system of equations is solved using Gaussian elimination.

The solution contains 98 dependent bits and 30 parameter bits. We divide the set
of parameter bits into 8 groups - 4 bits each, with a smaller top bit group (2 bits).
For each bit combination within a group, state bit flip masks are calculated such that
when xored with the initial solution state, the dependent variables are set to the 
correct values for a given combination of parameter values. 

Each GPU thread iterates through its assigned group of solutions and checks conditions
8-11 going forward. Matching states are saved in a GPU local buffer.

Finaly, the forward filter takes all candidate states found so far and checks the remaining
conditions in the enchantWithLevels pipeline. Matching states are saved as output.
*/


// Equation set generator - uses the provided seed to determine the positioning
// of the set bits in the addition columns.

Equation* generate_equations(int num_calls, int state_offset, uint64_t equation_seed) {
    return NULL;
}

// Bitmask set generator
// ...

// GPU Kernel
// ...