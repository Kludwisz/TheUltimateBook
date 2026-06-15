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
    int eq_count = num_calls * 13;
    Equation* eqns = malloc(eq_count * sizeof(Equation));
    if (eqns == NULL) return NULL;

    FXTMatrix fxtm = { 0 }, fxtmTransposed = { 0 };

    int permutation[50] = {0};
    for (int i = 0; equation_seed > 0; i++) {
        permutation[i] = equation_seed % 3;
        equation_seed /= 3;
    }

    int eq_id = 0;
    int pm_id = 0;
    for (int c=state_offset, cnt=0; cnt < num_calls; cnt++, c+=2) {
        /* old approach (14 eqns):
            .......31 30 29 28 27 26 25 24..............
            ........p..0..0..0..1..0..0................. LO
            ........q..a..b..c..d..0..0................. ROTL17(LO)
        +   ........r..A..B..C..D..0.................... ROTL17(HI)
        -------------------------------------------------
            ........0..0..0..0..0.......................
        */
        /* new approach (13 eqns):
            .......31 30 29 28 27 26 25 24..............
            ........p..0..0..0..0..1..0................. LO
            ........q..a..b..c..d..e..0................. ROTL17(LO)
        +   ........r..A..B..C..D..E.................... ROTL17(HI)
        -------------------------------------------------
            ........0..0..0..0..0.......................
        */

        DEBUG("reached c = %d, pm_id = %d, eq_id = %d\n", c, pm_id, eq_id);

        // 5 fixed-bit equations for cols 30-26 with permutations
        int perms_30_27[5];
        for (int i = 0; i < 5; i++, pm_id++) {
            perms_30_27[i] = permutation[pm_id];
        }
        eqns[eq_id++] = get_state_bit_equation(c, perms_30_27[0], 30, 0);
        eqns[eq_id++] = get_state_bit_equation(c, perms_30_27[1], 29, 0);
        eqns[eq_id++] = get_state_bit_equation(c, perms_30_27[2], 28, 0);
        eqns[eq_id++] = get_state_bit_equation(c, perms_30_27[3], 27, 0);
        eqns[eq_id++] = get_state_bit_equation(c, perms_30_27[4], 26, 1);

        // 5 aA pair equations for cols 30-26
        for (int bit = 30, i = 0; bit >= 26; bit--, i++) {
            Equation col_eq;
            initEquation(&col_eq, 0, 0, 1); // note the 1 on the RHS here
            for (int p = 0; p < 3; p++) {
                if (p == perms_30_27[i]) continue;
                Equation other_eq = get_state_bit_equation(c, p, bit, 0);
                xorEquationBy(&col_eq, &other_eq);
            }
            eqns[eq_id++] = col_eq;
        }

        // col 25 has two zeros to reduce max carry - 2 equations
        eqns[eq_id++] = get_state_bit_equation(c, XORO_LOW, 25, 0);
        eqns[eq_id++] = get_state_bit_equation(c, XORO_LOW_ROTL, 25, 0);

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

typedef struct GroupMaskSet GroupMaskSet;
struct GroupMaskSet {
    //        block           iter    thread
    // [2b][2b][2b][2b][2b] | [6b] | [4b][4b]
    //
    // block & thread masks - device memory
    // iter masks - constant memory - each thread accesses the exact same elements

    uint64_t MASK_BLOCK[5][4 * 2];
    uint64_t MASK_THREAD[2][16 * 2];
    uint64_t MASK_ITER[64 * 2];
};

void flip_bit(uint64_t *mask_lo, uint64_t *mask_hi, int state_bit_index) {
    if (state_bit_index >= 64) {
        *mask_hi ^= (1ULL << (state_bit_index - 64));
    }
    else {
        *mask_lo ^= (1ULL << state_bit_index);
    }
}

int is_param_set(Equation* eqn, int param_idx) {
    uint64_t mask = eqn->lhs[param_idx / 64];
    int shift = (63 - (param_idx % 64));
    if (mask & (1ULL << shift)) {
        return 1;
    }
    return 0;
}

void fill_mask_group(uint64_t *mask_group, uint64_t *per_param_masks, int param_idx_min_incl, int param_idx_max_incl) {
    uint32_t num_params = param_idx_max_incl - param_idx_min_incl + 1;
    uint32_t max_val = 1u << (num_params - 1);

    for (uint32_t p = 0; p < max_val; p++) {
        uint32_t pvals = p;
        uint64_t combined_lo = 0, combined_hi = 0;
        for (int idx = param_idx_min_incl; idx <= param_idx_max_incl; idx++) {
            uint32_t val = pvals & 1;
            pvals >>= 1;
            if (val) {
                combined_lo ^= per_param_masks[idx * 2];
                combined_hi ^= per_param_masks[idx * 2 + 1];
            }
        }

        // TODO save the mask!
    }
}

void generate_masks_for_solution(const Solution *sol, GroupMaskSet *gms) {
    uint64_t per_param_masks[24 * 2];
    
    // firstly, calculate masks for individual parameters
    for (int param_idx = 0, bit_idx = 0; param_idx < sol->parameterCount && bit_idx < 128; param_idx++) {
        while (!(sol->isParameter[bit_idx])) {
            bit_idx++;
        }
        uint64_t mask_lo = 0, mask_hi = 0;
        flip_bit(&mask_lo, &mask_hi, stateBitIndexOf(bit_idx));

        // for each equation where the parameter bit is set, flip the 
        // dependent variable determined by that equation in the state mask
        for (int eID = 0; eID < sol->equationCount; eID++) {
            if (is_param_set(&(sol->equations[eID]), bit_idx)) {
                flip_bit(&mask_lo, &mask_hi, stateBitIndexOf(sol->dependentVar[eID]));
            }
        }

        per_param_masks[param_idx * 2] = mask_lo;
        per_param_masks[param_idx * 2 + 1] = mask_hi;
        bit_idx++;
    }

    // now we can combine the per-param masks and calculate "aggregated"
    // flip masks for the parameter combinations defined in GroupMaskSet.

    // TODO
    for (int block_group_idx = 0; block_group_idx < 5; block_group_idx++) {

    }

    for (int thread_group_idx = 0; thread_group_idx < 2; thread_group_idx++) {

    }

    for (int iter_idx = 0; iter_idx < 64; iter_idx++) {
        int mask = iter_idx;
    }
}

// ---------------------------------------
// GPU Kernel
// ...

// ---------------------------------------
// Host-side code

void printXoro(const Xoroshiro* state)
{
    printf("Current Xoroshiro state:   ( %llu, %llu )\n", state->lo, state->hi);
}

int test_solver() {
    Solution* sol = generate_starting_solution(8, 5*2, 2 + 3);

    Xoroshiro states[8];
    getAllXoroshiroStates(sol, states, 8);

    Xoroshiro s = {states[1].lo, states[1].hi};
    printXoro(&s);
    for (int i = 0; i < 10; i++)
        xNextLong(&s);
    for (int i = 0; i < 8; i++) {
        //testNextInt(&s, 50);
        //printf("xNextLong() = %08x\n", (unsigned int)(xNextLong(&s) & (0xFFFFFFFFu)));
        printf("nextInt(50) = %d\n", xNextInt(&s, 50));
        xNextLong(&s);
    }
    
    free(sol);
    return 0;
}

int test_group_masks() {
    Solution* sol = generate_starting_solution(8, 5*2, 2 + 3);
    Xoroshiro startState;
    getAllXoroshiroStates(sol, &startState, 1);

    GroupMaskSet gms;
    generate_masks_for_solution(sol, &gms);
    free(sol);

    // TODO check if it's possible to use the group masks
}

int main() {
    //return test_solver();
    return test_group_masks();

    //Solution* sol = generate_starting_solution(7, 5*2, 2 + 3);
}
