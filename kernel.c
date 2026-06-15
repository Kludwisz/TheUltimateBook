#include "state_solver.h"
#include "xoro_matrix.h"
#include "utils.h"
#include <stdlib.h>
#include <stdio.h>
static const bool DEBUG_MODE = true;

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

// ---------------------------------------
// Equation set generator - uses the provided seed to determine the positioning
// of the set bits in the addition columns.

#define XORO_LOW_ROTL (2)
#define XORO_HIGH_ROTL (1)
#define XORO_LOW (0)

Equation get_state_bit_equation(int stateJump, int statePart, int bitIndex, int bitValue) {
    FXTMatrix fxtm = { 0 }, fxtmTransposed = { 0 };
    fastXoroMatrixPower(&XOROSHIRO_STANDARD_FXTM, stateJump, &fxtm);
    transposeFXTM(&fxtm, &fxtmTransposed);

    if (statePart == XORO_HIGH_ROTL || statePart == XORO_LOW_ROTL) {
        bitIndex = (bitIndex - 17 + 64) % 64;
    }

    Equation eqn;
    initEquation(&eqn, 
        fxtmTransposed.M[0][statePart & 1][63-bitIndex], 
        fxtmTransposed.M[1][statePart & 1][63-bitIndex], 
        bitValue);
    return eqn;
}

Solution* generate_starting_solution(int num_calls, int state_offset, uint64_t equation_seed) {
    int eq_count = num_calls * 14;
    Equation* eqns = malloc(eq_count * sizeof(Equation));
    if (eqns == NULL) return NULL;

    FXTMatrix fxtm = { 0 }, fxtmTransposed = { 0 };

    int permutation[40];
    for (int i = 0; equation_seed > 0; i++) {
        permutation[i] = equation_seed % 3;
        equation_seed /= 3;
    }

    int eq_id = 0;
    int pm_id = 0;
    for (int c=state_offset, cnt=0; cnt < num_calls; cnt++, c+=2) {
        /*
            .......31 30 29 28 27 26 25 24..............
            ........p..0..0..0..1..0..0................. LO
            ........q..a..b..c..d..0..0................. ROTL17(LO)
        +   ........r..A..B..C..D..0.................... ROTL17(HI)
        -------------------------------------------------
            ........0..0..0..0..0.......................
        */

        DEBUG("reached c = %d, pm_id = %d, eq_id = %d\n", c, pm_id, eq_id);

        // 4 fixed-bit equations for cols 30-27 with permutations
        int perms_30_27[4];
        for (int i = 0; i < 4; i++, pm_id++) {
            perms_30_27[i] = permutation[pm_id++];
        }
        eqns[eq_id++] = get_state_bit_equation(c, perms_30_27[0], 30, 0);
        eqns[eq_id++] = get_state_bit_equation(c, perms_30_27[1], 29, 0);
        eqns[eq_id++] = get_state_bit_equation(c, perms_30_27[2], 28, 0);
        eqns[eq_id++] = get_state_bit_equation(c, perms_30_27[3], 27, 1);

        // 4 aA pair equations for cols 30-27
        for (int bit = 30, i = 0; bit >= 27; bit--, i++) {
            Equation col_eq;
            initEquation(&col_eq, 0, 0, 1); // note the 1 on the RHS here
            for (int p = 0; p < 3; p++) {
                if (p == perms_30_27[i]) continue;
                Equation other_eq = get_state_bit_equation(c, p, bit, 0);
                xorEquationBy(&col_eq, &other_eq);
            }
            eqns[eq_id++] = col_eq;
        }

        // column 26 is all zeros
        eqns[eq_id++] = get_state_bit_equation(c, XORO_LOW, 26, 0);
        eqns[eq_id++] = get_state_bit_equation(c, XORO_LOW_ROTL, 26, 0);
        eqns[eq_id++] = get_state_bit_equation(c, XORO_HIGH_ROTL, 26, 0);

        // col 25 also uses a permutation but in a "negative" way
        int perm_col_25 = permutation[pm_id++]; 
        for (int p = 0; p < 3; p++) {
            if (p == perm_col_25) continue;
            eqns[eq_id++] = get_state_bit_equation(c, p, 25, 0);
        }

        // 1 pqr equation for col 31
        Equation eq_p = get_state_bit_equation(c, XORO_LOW_ROTL, 31, 1); // also note the 1 on the RHS here
        Equation eq_q = get_state_bit_equation(c, XORO_HIGH_ROTL, 31, 0);
        Equation eq_r = get_state_bit_equation(c, XORO_LOW, 31, 0);
        xorEquationBy(&eq_p, &eq_q);
        xorEquationBy(&eq_p, &eq_r);
        eqns[eq_id++] = eq_p;
    }
    DEBUG("reached solution gen\n");

    Solution *out = (Solution *)malloc(sizeof(Solution));
    gaussianElimGF2(eqns, eq_count, out);
    free(eqns);

    DEBUG("solution gen succeeded\n");
    return out;
}

// ---------------------------------------
// Bitmask set generator
// ...

// ---------------------------------------
// GPU Kernel
// ...

// ---------------------------------------
// Host-side code

void printXoro(const Xoroshiro* state)
{
    printf("Current Xoroshiro state:   ( %llu, %llu )\n", state->lo, state->hi);
}

int main() {
    Solution* sol = generate_starting_solution(7, 5*2, 0);

    Xoroshiro states[8];
    DEBUG("a");
    getAllXoroshiroStates(sol, states, 8);
    DEBUG("b");

    Xoroshiro s = {states[1].lo, states[1].hi};
    printXoro(&s);
    xNextLong(&s);
    for (int i = 0; i < 7; i++) {
        //testNextInt(&s, 50);
        //printf("xNextLong() = %08x\n", (unsigned int)(xNextLong(&s) & (0xFFFFFFFFu)));
        printf("nextInt(50) = %d\n", xNextInt(&s, 50));
    }
    
    free(sol);
    return 0;
}