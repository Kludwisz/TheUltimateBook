#include "state_solver.h"
#include "xoro_matrix.h"
#include <cstdlib>
#include <cstdio>

#define XORO_HIGH (1)
#define XORO_LOW (0)

#define ROTL_17(bitIndex) ((bitIndex - 17 + 64) % 64)

Equation get_state_bit_equation(int stateJump, int statePart, int bitIndex, int bitValue) {
    FXTMatrix fxtm = { 0 }, fxtmTransposed = { 0 };
    fastXoroMatrixPower(&XOROSHIRO_STANDARD_FXTM, stateJump, &fxtm);
    transposeFXTM(&fxtm, &fxtmTransposed);

    Equation eqn;
    initEquation(&eqn, fxtmTransposed.M[0][statePart][63-bitIndex], fxtmTransposed.M[1][statePart][63-bitIndex], bitValue);
    return eqn;
}

/**
 * Returns a Xoroshiro state solution for a sequence of nextInt constraints
 */
void solve_for_consecutive_nextInt_bits(int num_calls, Solution *out) {
    int eq_count = num_calls * 14;
    
    Equation* eqns = static_cast<Equation *>(malloc(eq_count * sizeof(Equation)));
    if (eqns == NULL) return;

    FXTMatrix fxtm = { 0 }, fxtmTransposed = { 0 };

    int eq_id = 0;
    for (int c = 1; c <= num_calls; c++) {
        /*
            .......31 30 29 28 27 26 25 24..............
            ........p..0..0..0..1..0..0................. LO
            ........q..a..b..c..d..0..0................. ROTL17(LO)
        +   ........r..A..B..C..D..0.................... ROTL17(HI)
        -------------------------------------------------
            ........0..0..0..0..0.......................
        */

        // 9 fixed-bit equations (no permutations just to test)
        eqns[eq_id++] = get_state_bit_equation(c, XORO_LOW, 30, 0);
        eqns[eq_id++] = get_state_bit_equation(c, XORO_LOW, 29, 0);
        eqns[eq_id++] = get_state_bit_equation(c, XORO_LOW, 28, 0);
        eqns[eq_id++] = get_state_bit_equation(c, XORO_LOW, 27, 1);

        eqns[eq_id++] = get_state_bit_equation(c, XORO_LOW, 26, 0);
        eqns[eq_id++] = get_state_bit_equation(c, XORO_LOW, 25, 0);
        eqns[eq_id++] = get_state_bit_equation(c, XORO_LOW, ROTL_17(26), 0);
        eqns[eq_id++] = get_state_bit_equation(c, XORO_LOW, ROTL_17(25), 0);
        eqns[eq_id++] = get_state_bit_equation(c, XORO_HIGH, ROTL_17(26), 0);

        // 4 aA pair equations
        for (int bit = 30; bit >= 27; bit--) {
            Equation eq_a = get_state_bit_equation(c, XORO_HIGH, ROTL_17(bit), 1); // note the 1 on the RHS here
            Equation eq_A = get_state_bit_equation(c, XORO_LOW, ROTL_17(bit), 0);
            xorEquationBy(&eq_a, &eq_A);
            eqns[eq_id++] = eq_a;
        }

        // 1 pqr equation
        {
            Equation eq_p = get_state_bit_equation(c, XORO_LOW, ROTL_17(31), 1); // also note the 1 on the RHS here
            Equation eq_q = get_state_bit_equation(c, XORO_HIGH, ROTL_17(31), 0);
            Equation eq_r = get_state_bit_equation(c, XORO_LOW, 31, 0);
            xorEquationBy(&eq_p, &eq_q);
            xorEquationBy(&eq_p, &eq_r);
            eqns[eq_id++] = eq_p;
        }
    }
    printf("%d\n", gaussianElimGF2(eqns, eq_count, out));

    printf("initial eq count: %d\n", eq_count);
    printf("sol: %d %d %d\n", out->isContradictory, out->knownVariableCount, out->parameterCount);

    printf("eq1: %016llx-%016llx  = %d\n", out->equations[0].lhs[0], out->equations[0].lhs[1], out->equations[0].rhs);
    printf("eq2: %016llx-%016llx  = %d\n", out->equations[1].lhs[0], out->equations[1].lhs[1], out->equations[1].rhs);
    printf("eq3: %016llx-%016llx  = %d\n", out->equations[2].lhs[0], out->equations[2].lhs[1], out->equations[2].rhs);

    free(eqns);
}

void printXoro(const Xoroshiro* state)
{
    printf("Current Xoroshiro state:   ( %llu, %llu )\n", state->lo, state->hi);
}


static inline int testNextInt(Xoroshiro *xr, uint32_t n)
{
    uint64_t val = xNextLong(xr);
    uint64_t r = (val & 0xFFFFFFFF) * n;
    if ((uint32_t)r < n)
    {
        printf("edge case, we'll ignore this.\n");
        while ((uint32_t)r < (~n + 1) % n)
        {
            r = (xNextLong(xr) & 0xFFFFFFFF) * n;
        }
    }
    printf("val = %08x, ret = %d\n", (val & 0xFFFFFFFF), r>>32);

    return r >> 32;
}


int main() {
    Solution sol;
    int outputs = 9;
    solve_for_consecutive_nextInt_bits(outputs, &sol);
    
    Xoroshiro states[8];
    getAllXoroshiroStates(&sol, states, 8);

    Xoroshiro s = {states[1].lo, states[1].hi};
    printXoro(&s);
    xNextLong(&s);
    for (int i = 0; i < outputs; i++) {
        testNextInt(&s, 50);
        //printf("xNextLong() = %08x\n", (unsigned int)(xNextLong(&s) & (0xFFFFFFFFu)));
        //printf("nextInt(50) = %d\n", xNextInt(&s, 50));
    }
    
    return 0;
}