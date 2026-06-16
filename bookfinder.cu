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

#include <chrono>
#include <cstring>
#include <cstdint>
#include <cstdlib>
#include <cstdio>
#include <unordered_set>

#include "cuda_utils.cuh"
#include "xoro_matrix.h"
#include "state_solver.h"

// Algorithm parameters
constexpr std::uint32_t NUM_REVERSED_STATES = 8;
constexpr std::uint32_t STATE_JUMP_SIZE = 5;
constexpr std::uint32_t ITER_MASK_BITS = 6;
constexpr std::uint32_t THREAD_MASK_BITS = 4;
constexpr std::uint32_t THREAD_MASK_GROUPS = 2;
constexpr std::uint32_t BLOCK_MASK_BITS = 2;
constexpr std::uint32_t BLOCK_MASK_GROUPS = 5;

constexpr std::uint32_t THREADS_PER_BLOCK = 256;


// safeguards
constexpr std::uint32_t TOTAL_PARAM_BITS = ITER_MASK_BITS 
        + THREAD_MASK_BITS * THREAD_MASK_GROUPS 
        + BLOCK_MASK_BITS * BLOCK_MASK_GROUPS;
constexpr std::uint32_t EXPECTED_PARAM_BITS = 128 - 13 * NUM_REVERSED_STATES;

static_assert(TOTAL_PARAM_BITS == EXPECTED_PARAM_BITS, "Invalid total parameter bits, review the input data.");
static_assert(THREADS_PER_BLOCK == (1U << (THREAD_MASK_BITS * THREAD_MASK_GROUPS)), "The thread mask size must match THREADS_PER_BLOCK.");

// --------------------------------------------------------------------------------
// GPU side code: just go brrrrrr and crunch as many Xoroshiro states as possible.

// kernel input data


constexpr std::uint32_t N_ITER_MASKS = (1U << ITER_MASK_BITS) * 2;
constexpr std::uint32_t N_BLOCK_MASKS = BLOCK_MASK_GROUPS * (1U << BLOCK_MASK_BITS) * 2;
constexpr std::uint32_t N_THREAD_MASKS = THREAD_MASK_GROUPS * (1U << THREAD_MASK_BITS) * 2;
__device__   std::uint64_t THREAD_MASKS[N_THREAD_MASKS];
__device__   std::uint64_t BLOCK_MASKS[N_BLOCK_MASKS];
__constant__ std::uint64_t ITER_MASKS[N_ITER_MASKS];

constexpr std::uint32_t MASK_I = (1U << ITER_MASK_BITS) - 1;
constexpr std::uint32_t MASK_T = (1U << THREAD_MASK_BITS) - 1;
constexpr std::uint32_t MASK_B = (1U << BLOCK_MASK_BITS) - 1;
constexpr std::uint32_t GROUP_I = (1U << ITER_MASK_BITS) * 2;
constexpr std::uint32_t GROUP_T = (1U << THREAD_MASK_BITS) * 2;
constexpr std::uint32_t GROUP_B = (1U << BLOCK_MASK_BITS) * 2;

// kernel output data
constexpr std::uint32_t N_RESULTS_MAX = 128 * 1024;
__managed__ Xoroshiro results[N_RESULTS_MAX];
__managed__ uint32_t result_count = 0;

// the kernel
__global__ void gaussianElimIterKernel(Xoroshiro initialState) {
    // per-thread computations
    std::uint32_t tid = threadIdx.x;
    initialState.lo ^= THREAD_MASKS[(tid & MASK_T) * 2];
    initialState.hi ^= THREAD_MASKS[(tid & MASK_T) * 2 + 1];
    tid >>= 4;
    initialState.lo ^= THREAD_MASKS[GROUP_T + (tid & MASK_T) * 2];
    initialState.hi ^= THREAD_MASKS[GROUP_T + (tid & MASK_T) * 2 + 1];

    // per-block computations
    std::uint32_t bid = blockIdx.x;
    #pragma unroll
    for (int i = 0; i < BLOCK_MASK_GROUPS; i++) {
        initialState.lo ^= BLOCK_MASKS[i*GROUP_B + (bid & MASK_B) * 2];
        initialState.hi ^= BLOCK_MASKS[i*GROUP_B + (bid & MASK_B) * 2 + 1];
        bid >>= 2;
    }

    // We have the initial state. Now all remaining state jump operations will be
    // handled using fully precomputed mask values stored in __constant__ memory. 

    for (int iterIndex = 0; iterIndex <= MASK_I; iterIndex++) {
        //printf("thread %d: %llu\n", threadIdx.x, ITER_MASKS[iterIndex*2]);
        Xoroshiro state = initialState;
        state.lo ^= ITER_MASKS[iterIndex*2];
        state.hi ^= ITER_MASKS[iterIndex*2 + 1];

        bool next_iter = false;
        #pragma unroll
        for (int j = 0; j < STATE_JUMP_SIZE; j++) {
            if (xNextIntFast(&state, 50) != 0) {
                next_iter = true;
                break; // filter
            }
            (void)xNextLong(&state); // skip 1
        }

        if (next_iter) {
            continue;
        }

        {   // got a partial result
            Xoroshiro out = initialState;
            out.lo ^= ITER_MASKS[iterIndex*2];
            out.hi ^= ITER_MASKS[iterIndex*2 + 1];

            if (result_count < N_RESULTS_MAX) {
                int idx = atomicAdd(&result_count, 1);
                results[idx] = out;
            }
        }
    }
}


// -------------------------------------------------------------------------------
// Host side code: parse input args, solve the system of equations, load the 
// correct data into GPU constant memory, print the results.

#define XORO_LOW_ROTL (2)
#define XORO_HIGH_ROTL (1)
#define XORO_LOW (0)

// Equation set generator - uses the provided seed to determine the positioning
// of the set bits in the addition columns.

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
    Equation* eqns = static_cast<Equation*>(malloc(eq_count * sizeof(Equation)));
    if (eqns == NULL) return NULL;

    int permutation[64] = {0};
    for (int i = 0; equation_seed > 0; i++) {
        permutation[i] = equation_seed % 3;
        equation_seed /= 3;
    }

    int eq_id = 0;
    int pm_id = 0;
    for (int c=state_offset, cnt=0; cnt < num_calls; cnt++, c+=2) {
        /*
            .......31 30 29 28 27 26 25 24..............
            ........p..0..0..0..0..1..0................. LO
            ........q..a..b..c..d..e..0................. ROTL17(LO)
        +   ........r..A..B..C..D..E.................... ROTL17(HI)
        -------------------------------------------------
            ........0..0..0..0..0.......................
        */

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

    Solution *out = static_cast<Solution *>(malloc(sizeof(Solution)));
    gaussianElimGF2(eqns, eq_count, out);
    free(eqns);

    return out;
}

// ---------------------------------------
// Bitmask set generator

typedef struct GroupMaskSet GroupMaskSet;
struct GroupMaskSet {
    // block & thread masks - device memory
    // iter masks - constant memory - each thread accesses the exact same elements
    uint64_t MASK_BLOCK[BLOCK_MASK_GROUPS][GROUP_B];
    uint64_t MASK_THREAD[THREAD_MASK_GROUPS][GROUP_T];
    uint64_t MASK_ITER[GROUP_I];
};

void flip_bit(uint64_t *mask_lo, uint64_t *mask_hi, int state_bit_index) {
    if (state_bit_index >= 64) {
        *mask_hi ^= (1ULL << (state_bit_index - 64));
    }
    else {
        *mask_lo ^= (1ULL << state_bit_index);
    }
}

int is_param_set(const Equation* eqn, int param_idx) {
    uint64_t mask = eqn->lhs[param_idx / 64];
    int shift = (63 - (param_idx % 64));
    if (mask & (1ULL << shift)) {
        return 1;
    }
    return 0;
}

void fill_mask_group(uint64_t mask_group[], uint64_t per_param_masks[], int param_idx_min_incl, int param_idx_max_incl) {
    uint32_t num_params = param_idx_max_incl - param_idx_min_incl + 1;
    uint32_t max_val = 1u << num_params;

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

        mask_group[p * 2] = combined_lo;
        mask_group[p * 2 + 1] = combined_hi; 
    }
}

void generate_masks_for_solution(const Solution *sol, GroupMaskSet *gms) {
    uint64_t per_param_masks[TOTAL_PARAM_BITS * 2];
    
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

    for (int block_group_idx = 0; block_group_idx < BLOCK_MASK_GROUPS; block_group_idx++) {
        fill_mask_group(gms->MASK_BLOCK[block_group_idx], per_param_masks, 
            0 + BLOCK_MASK_BITS*block_group_idx, 
            BLOCK_MASK_BITS - 1 + BLOCK_MASK_BITS * block_group_idx
        );
    }

    constexpr std::uint32_t min_iter_bit = 2 + BLOCK_MASK_BITS * (BLOCK_MASK_GROUPS - 1);
    constexpr std::uint32_t max_iter_bit = min_iter_bit + ITER_MASK_BITS - 1;
    fill_mask_group(gms->MASK_ITER, per_param_masks, min_iter_bit, max_iter_bit);

    constexpr std::uint32_t min_thread_bit = max_iter_bit + 1;
    constexpr std::uint32_t max_thread_bit = min_thread_bit + THREAD_MASK_BITS - 1;
    for (int thread_group_idx = 0; thread_group_idx < THREAD_MASK_GROUPS; thread_group_idx++) {
        fill_mask_group(gms->MASK_THREAD[thread_group_idx], per_param_masks, 
            min_thread_bit + THREAD_MASK_BITS * thread_group_idx, 
            max_thread_bit + THREAD_MASK_BITS * thread_group_idx
        );
    }
}


void setup_search(std::uint64_t eqn_seed, Xoroshiro *initialState) {
    // Generate masks & initial state
    Solution *sol = generate_starting_solution(NUM_REVERSED_STATES, STATE_JUMP_SIZE*2, eqn_seed);
    getAllXoroshiroStates(sol, initialState, 1);
    GroupMaskSet gms;
    generate_masks_for_solution(sol, &gms);
    free(sol);

    // Copy data to device and constant mem
    CUDA_CHECK(cudaMemcpyToSymbol(ITER_MASKS, gms.MASK_ITER, 128 * sizeof(uint64_t)));

    uint64_t BLOCK_MASKS_h[N_BLOCK_MASKS];
    uint64_t THREAD_MASKS_h[N_THREAD_MASKS];
    for (int i = 0; i < BLOCK_MASK_GROUPS; i++) {
        for (int j = 0; j < GROUP_B; j++) {
            BLOCK_MASKS_h[i * GROUP_B + j] = gms.MASK_BLOCK[i][j];
        }
    }
    for (int i = 0; i < THREAD_MASK_GROUPS; i++) {
        for (int j = 0; j < GROUP_T; j++) {
            THREAD_MASKS_h[i * GROUP_T + j] = gms.MASK_THREAD[i][j];
        }
    }
    CUDA_CHECK(cudaMemcpyToSymbol(BLOCK_MASKS, BLOCK_MASKS_h, N_BLOCK_MASKS * sizeof(uint64_t)));
    CUDA_CHECK(cudaMemcpyToSymbol(THREAD_MASKS, THREAD_MASKS_h, N_THREAD_MASKS * sizeof(uint64_t)));
}

void run(std::uint64_t eqn_seed) {
    Xoroshiro initialState;
    setup_search(eqn_seed, &initialState);
    
    result_count = 0;

    constexpr std::uint32_t N_BLOCKS = (1U << (TOTAL_PARAM_BITS - ITER_MASK_BITS)) / THREADS_PER_BLOCK;
    gaussianElimIterKernel <<< N_BLOCKS, THREADS_PER_BLOCK >>> (initialState);
    CUDA_CHECK(cudaDeviceSynchronize());
}

int main(int argc, char **argv) {
    std::uint64_t start_eqn_seed = 0;
    std::uint32_t num_runs = 1;

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--seed") == 0) {
            if (i == argc-1 || sscanf(argv[i+1], "%" PRIu64, &start_eqn_seed) != 1) {
                std::fprintf(stderr, "Illegal or unspecified starting seed.\n");
                return 1;
            }
        }
        else if (strcmp(argv[i], "--runs") == 0) {
            if (i == argc-1 || sscanf(argv[i+1], "%" PRIu32, &num_runs) != 1) {
                std::fprintf(stderr, "Illegal or unspecified run count.\n");
                return 1;
            }
        }
    }

    std::fprintf(stderr, "Running search, seed=%" PRIu64 ", runs=%" PRIu32 "\n", start_eqn_seed, num_runs);

    std::chrono::steady_clock::time_point t0 = std::chrono::steady_clock::now(), t1;
    std::uint64_t eqn_seed = start_eqn_seed;
    std::unordered_set<Xoroshiro, XoroshiroHash> solutions;

    for (int r = 0; r < num_runs; r++) {
        run(eqn_seed);

        if (r % 10 == 0) {
            t0 = std::chrono::steady_clock::now();
        }
        if (r % 10 == 9) {
            t1 = std::chrono::steady_clock::now();
            const auto int_ms = std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count();
            std::fprintf(stderr, "run %d/%d:  %lld ms\n", r+1, static_cast<int>(num_runs), int_ms);
        }

        for (std::uint32_t i = 0; i < result_count; i++) {
            if (solutions.find(results[i]) == solutions.end()) {
                std::fprintf(stdout, "%" PRIu64 " %" PRIu64 "\n", results[i].lo, results[i].hi);
                solutions.insert(results[i]);

                Xoroshiro xr = results[i];
                for (int i = 0; i < NUM_REVERSED_STATES + STATE_JUMP_SIZE; i++) { 
                    std::fprintf(stderr, "nextInt(50) = %d\n", xNextInt(&xr, 50));
                    xNextLong(&xr);
                }
            }
        }
        if (r % 20 == 0) {
            Xoroshiro next;
            xSetSeed(&next, eqn_seed);
            eqn_seed = xNextLongJ(&next);
        }
        else {
            eqn_seed += 1;
        }
    }
}