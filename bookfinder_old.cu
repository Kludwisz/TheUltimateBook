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

The solution contains 104 dependent bits and 24 parameter bits. We divide the set
of parameter bits into smaller groups - block groups, thread groups, and the iteration group.
For each bit combination within a group, state bit flip masks are calculated such that
when xored with the initial solution state, the dependent variables are set to the 
correct values for a given combination of parameter values. 

Each GPU thread iterates through its assigned group of solutions and checks the remaining conditions
going forward. Matching states are saved in a GPU local buffer.

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

__device__ __forceinline__ bool process(Xoroshiro state, uint64_t lo, uint64_t hi) {
    state.lo ^= lo;
    state.hi ^= hi;

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
        return false;
    }

    return true;
}

// the kernel
__global__ void gaussianElimIterKernel(Xoroshiro initialState) {
    // per-thread computations
    std::uint32_t tid = threadIdx.x;

    ulonglong2 data0 = reinterpret_cast<ulonglong2 *>(THREAD_MASKS)[(tid & MASK_T)];
    initialState.lo ^= data0.x;
    initialState.hi ^= data0.y;
    tid >>= 4;
    ulonglong2 data1 = reinterpret_cast<ulonglong2 *>(THREAD_MASKS)[(GROUP_T>>1) + (tid & MASK_T)];
    initialState.lo ^= data1.x;
    initialState.hi ^= data1.y;

    // per-block computations
    std::uint32_t bid = blockIdx.x;
    #pragma unroll
    for (int i = 0; i < BLOCK_MASK_GROUPS; i++) {
        ulonglong2 data = reinterpret_cast<ulonglong2 *>(BLOCK_MASKS)[i*(GROUP_B>>1) + (bid & MASK_B)];

        initialState.lo ^= data.x;
        initialState.hi ^= data.y;
        bid >>= 2;
    }

    // We have the initial state. Now all remaining state jump operations will be
    // handled using fully precomputed mask values stored in __constant__ memory. 
    #pragma unroll
    for (int iterIndex = 0; iterIndex <= MASK_I >> 1; iterIndex++) {
        ulonglong4 data = reinterpret_cast<ulonglong4 *>(ITER_MASKS)[iterIndex];
        if (process(initialState, data.x, data.y)) {   // got a partial result
            Xoroshiro out = initialState;
            out.lo ^= data.x;
            out.hi ^= data.y;

            if (result_count < N_RESULTS_MAX) {
                int idx = atomicAdd(&result_count, 1);
                results[idx] = out;
            }
        }
        if (process(initialState, data.z, data.w)) {  
            Xoroshiro out = initialState;
            out.lo ^= data.z;
            out.hi ^= data.w;

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

constexpr std::uint32_t NUM_ALL_ZERO_STATES = 13;
static Equation PRECOMPUTED_EQNS[32][3][64][2] = {0};

void precompute_equations() {
    for (std::uint32_t state_jump = 0; state_jump < 32; state_jump++) {
        for (std::uint32_t state_part = 0; state_part < 3; state_part++) {
            for (std::uint32_t bit_index = 0; bit_index < 64; bit_index++) {
                for (std::uint32_t bit_value = 0; bit_value < 2; bit_value++) {
                    PRECOMPUTED_EQNS[state_jump][state_part][bit_index][bit_value]
                            = get_state_bit_equation(state_jump, state_part, bit_index, bit_value);
                }
            }
        }
    }
}

Equation get_precomputed_equation(int stateJump, int statePart, int bitIndex, int bitValue) {
    return PRECOMPUTED_EQNS[stateJump][statePart][bitIndex][bitValue];
}

void generate_state_seqn(int state_idxs[NUM_REVERSED_STATES], std::uint64_t eqn_seed) {
    std::uint64_t seqn_seed = 69420ull; //std::chrono::steady_clock::now().time_since_epoch().count() + eqn_seed;
    
    Xoroshiro seqn_rng;
    xSetSeed(&seqn_rng, seqn_seed);

    bool state_taken[NUM_ALL_ZERO_STATES] = {0};
    int count = 0;
    while (count < NUM_REVERSED_STATES) {
        int idx = xNextInt(&seqn_rng, NUM_ALL_ZERO_STATES);
        if (state_taken[idx]) {
            continue;
        }
        state_taken[idx] = true;
        count++;
    }

    int j = 0;
    for (int idx = 0; idx < NUM_ALL_ZERO_STATES; idx++) {
        if (state_taken[idx]) {
            state_idxs[j++] = idx;
        }
    }
}

Solution* generate_starting_solution(int num_calls, std::uint64_t equation_seed) {
    //timer_start();
    int eq_count = num_calls * 13;
    Equation* eqns = static_cast<Equation*>(malloc(eq_count * sizeof(Equation)));
    if (eqns == NULL) return NULL;

    int permutation[64] = {0};
    for (int i = 0; equation_seed > 0; i++) {
        permutation[i] = equation_seed % 3;
        equation_seed /= 3;
    }

    int state_idxs[NUM_REVERSED_STATES] = {0};
    generate_state_seqn(state_idxs, equation_seed);

    int eq_id = 0;
    int pm_id = 0;
    for (int sidx = 0; sidx < NUM_REVERSED_STATES; sidx++) {
        int c = state_idxs[sidx] * 2;
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
        eqns[eq_id++] = get_precomputed_equation(c, perms_30_27[0], 30, 0);
        eqns[eq_id++] = get_precomputed_equation(c, perms_30_27[1], 29, 0);
        eqns[eq_id++] = get_precomputed_equation(c, perms_30_27[2], 28, 0);
        eqns[eq_id++] = get_precomputed_equation(c, perms_30_27[3], 27, 0);
        eqns[eq_id++] = get_precomputed_equation(c, perms_30_27[4], 26, 1);

        // 5 aA pair equations for cols 30-26
        for (int bit = 30, i = 0; bit >= 26; bit--, i++) {
            Equation col_eq;
            initEquation(&col_eq, 0, 0, 1); // note the 1 on the RHS here
            for (int p = 0; p < 3; p++) {
                if (p == perms_30_27[i]) continue;
                Equation other_eq = get_precomputed_equation(c, p, bit, 0);
                xorEquationBy(&col_eq, &other_eq);
            }
            eqns[eq_id++] = col_eq;
        }

        // col 25 has two zeros to reduce max carry - 2 equations
        eqns[eq_id++] = get_precomputed_equation(c, XORO_LOW, 25, 0);
        eqns[eq_id++] = get_precomputed_equation(c, XORO_LOW_ROTL, 25, 0);

        // 1 pqr equation for col 31
        Equation eq_p = get_precomputed_equation(c, XORO_LOW_ROTL, 31, 1); // also note the 1 on the RHS here
        Equation eq_q = get_precomputed_equation(c, XORO_HIGH_ROTL, 31, 0);
        Equation eq_r = get_precomputed_equation(c, XORO_LOW, 31, 0);
        xorEquationBy(&eq_p, &eq_q);
        xorEquationBy(&eq_p, &eq_r);
        eqns[eq_id++] = eq_p;
    }
    //timer_end("|-- generate equations");

    //timer_start();
    Solution *out = static_cast<Solution *>(malloc(sizeof(Solution)));
    gaussianElimGF2(eqns, eq_count, out);
    free(eqns);
    //timer_end("|-- solve equations");

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

typedef unsigned __int128 uint128_t; 

uint64_t bool_array_to_uint64(const bool arr[64]) { 
    uint64_t result = 0;

    for (int i = 0; i < 64; i++) {
        if (arr[i]) {
            result |= ((uint64_t)1 << (63 - i));
        }
    }
    return result;
}

#define RUNS_PER_BATCH 256

struct Batch {
    Xoroshiro states[RUNS_PER_BATCH];
    uint64_t per_param_lo[RUNS_PER_BATCH][24];
    uint64_t per_param_hi[RUNS_PER_BATCH][24];
};

__global__ void kernel(Batch *batch) {
    uint32_t tid = threadIdx.x + blockDim.x * blockIdx.x;

    uint64_t lo_flip = 0, hi_flip = 0;
    uint32_t remaining = tid;
    
    for (int bit = 0; bit < 24; bit++) {
        if ((tid >> bit) & 1) {
            lo_flip ^= batch->per_param_lo[blockIdx.y][bit];
            hi_flip ^= batch->per_param_hi[blockIdx.y][bit];
        }
    }
    
    Xoroshiro initial = batch->states[blockIdx.y];

    Xoroshiro state = initial;
    state.lo ^= lo_flip;
    state.hi ^= hi_flip;

    for (int j = 0; j < NUM_ALL_ZERO_STATES; j++) {
        if (xNextIntFast(&state, 50) != 0) return;
        (void)xNextLong(&state);
    }

    if (result_count < N_RESULTS_MAX) {
        int idx = atomicAdd(&result_count, 1);
        initial.lo ^= lo_flip;
        initial.hi ^= hi_flip;
        results[idx] = initial; 
    }
}

#include <omp.h>

Batch generate_batch(uint64_t *eqn_seed) {
    Batch batch;

    uint64_t seeds[RUNS_PER_BATCH];
    seeds[0] = *eqn_seed;
    // printf("%lu\n", seeds[0]);
    for (int i = 1; i < RUNS_PER_BATCH; i++) {
        Xoroshiro tmp;
        xSetSeed(&tmp, seeds[i - 1]);
        seeds[i] = xNextLongJ(&tmp);
        // printf("%lu\n", seeds[i]);
    }

    #pragma omp parallel for schedule(static)
    for (int i = 0; i < RUNS_PER_BATCH; i++) {
        Solution *sol = generate_starting_solution(
            NUM_REVERSED_STATES,
            seeds[i]
        );

        getAllXoroshiroStates(sol, &batch.states[i], 1);

        int param_idx = 0;

        for (int bit_idx = 0; bit_idx < 128; bit_idx++) {
            if (!sol->isParameter[bit_idx])
                continue;

            uint64_t mask_lo = 0;
            uint64_t mask_hi = 0;

            flip_bit(&mask_lo, &mask_hi, stateBitIndexOf(bit_idx));

            for (int eID = 0; eID < sol->equationCount; eID++) {
                if (is_param_set(&sol->equations[eID], bit_idx)) {
                    flip_bit(
                        &mask_lo,
                        &mask_hi,
                        stateBitIndexOf(sol->dependentVar[eID])
                    );
                }
            }

            batch.per_param_lo[i][param_idx] = mask_lo;
            batch.per_param_hi[i][param_idx] = mask_hi;
            param_idx++;
        }

        free(sol);
    }
    *eqn_seed = seeds[RUNS_PER_BATCH - 1];

    Xoroshiro next;
    xSetSeed(&next, *eqn_seed);
    *eqn_seed = xNextLongJ(&next);

    return batch;
}

// Batch generate_batch(uint64_t *eqn_seed) {
//     Batch batch;
//     for (int i = 0; i < RUNS_PER_BATCH; i++) {
//         Solution *sol = generate_starting_solution(NUM_REVERSED_STATES, STATE_JUMP_SIZE*2, *eqn_seed);
//         getAllXoroshiroStates(sol, &batch.states[i], 1);

//         result_count = 0;

//         int param_idx = 0;
//         for (int bit_idx = 0; bit_idx < 128; bit_idx++) {
//             if (!sol->isParameter[bit_idx]) continue;
            
//             uint64_t mask_lo = 0, mask_hi = 0;
//             flip_bit(&mask_lo, &mask_hi, stateBitIndexOf(bit_idx));  

//             for (int eID = 0; eID < sol->equationCount; eID++) {
//                 if (is_param_set(&sol->equations[eID], bit_idx)) {
//                     flip_bit(&mask_lo, &mask_hi, stateBitIndexOf(sol->dependentVar[eID]));
//                 }
//             }

//             batch.per_param_lo[i][param_idx] = mask_lo;
//             batch.per_param_hi[i][param_idx] = mask_hi;
//             param_idx++;
//         }   
            
//         // cudaMemcpy(d_per_param_lo, per_param_lo, 24 * sizeof(uint64_t), cudaMemcpyHostToDevice);
//         // cudaMemcpy(d_per_param_hi, per_param_hi, 24 * sizeof(uint64_t), cudaMemcpyHostToDevice);
//         Xoroshiro next;
//         xSetSeed(&next, *eqn_seed);
//         (*eqn_seed) = xNextLongJ(&next);
//     }

//     return batch;
// }

#include <iostream>

uint64_t run(std::uint64_t eqn_seed, Batch *d_batch) {
    Batch batch = generate_batch(&eqn_seed);
    cudaMemcpy(d_batch, &batch, sizeof(Batch), cudaMemcpyHostToDevice);
    
    result_count = 0;

    dim3 blocks(65536, RUNS_PER_BATCH);
    kernel<<<blocks, 256>>>(d_batch);
    cudaDeviceSynchronize();
    return eqn_seed;
}

void full_check(Xoroshiro state7, std::uint32_t& full_result_counter) {
    // back 12 for the nextInt states, then back another (?) to undo the enchant_with_levels stuff
    Xoroshiro base_state = state7;

    // FULL_64, FULL_64 is 0
    FXTMatrix back = {0};
    fastXoroMatrixPower128(&XOROSHIRO_STANDARD_FXTM, FULL_64, FULL_64 - 20, &back);
    advanceXoroshiroFXTM(&base_state, &back);

    int max_enchants = 0;
    for (int i = 0; i < 40; i++) {
        Xoroshiro state = base_state;
        
        const std::uint32_t enchantability = 1;
        const std::uint32_t delta = enchantability / 4 + 1;

        std::uint32_t level = 31 + xNextInt(&state, delta) + xNextInt(&state, delta);
        float f1 = xNextFloat(&state);
        float f2 = xNextFloat(&state);
        float amplifier = (f1 + f2 - 1.0F) * 0.15F;
        level = std::floor( ((float)level + (float)level * amplifier) + 0.5F );
        (void)xNextLong(&state);

        int num_enchants = 1;
        while (xNextInt(&state, 50) <= level) {
            num_enchants++;
            (void)xNextLong(&state);
            level /= 2;
        }
        if (num_enchants > max_enchants) {
            max_enchants = num_enchants;
        }

        if (num_enchants >= 16) {
            full_result_counter++;
            std::fprintf(stderr, "GOOD SOLUTION!\n");
            std::fprintf(stderr, "%d %" PRIu64 " %" PRIu64 "\n", num_enchants, base_state.lo, base_state.hi);
            // std::fprintf(stdout, "%d %" PRIu64 " %" PRIu64 "\n", num_enchants, base_state.lo, base_state.hi);
        }
        (void)xNextLong(&base_state);
    }

    // std::fprintf(stderr, "Candidate sol max enchants: %d\n", max_enchants);
}

int main(int argc, char **argv) {
    std::uint64_t start_eqn_seed = 0;
    std::uint32_t num_runs = 1;
    std::uint32_t results_partial = 0, results_full = 0;

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

    std::fprintf(stderr, "Precomputing equations...\n");
    precompute_equations();

    std::fprintf(stderr, "Running search, seed=%" PRIu64 ", runs=%" PRIu32 "\n", start_eqn_seed, num_runs);

    std::chrono::steady_clock::time_point t0 = std::chrono::steady_clock::now(), t1;
    std::uint64_t eqn_seed = start_eqn_seed;
    std::unordered_set<Xoroshiro, XoroshiroHash> solutions;

    Batch *d_batch;
    cudaMalloc(&d_batch, sizeof(Batch));

    for (int r = 0; r < num_runs; r += RUNS_PER_BATCH) {
        eqn_seed = run(eqn_seed, d_batch);
        // run(eqn_seed);

        if (r % 50 == 0) {
            t0 = std::chrono::steady_clock::now();
        }
        if (r % 50 == 49) {
            t1 = std::chrono::steady_clock::now();
            const auto int_ms = std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count();
            std::fprintf(stderr, "--- run %d/%d:  %lld ms  (res: p=%" PRIu32 " f=%" PRIu32 ")\n", 
                r+1, static_cast<int>(num_runs), int_ms, results_partial, results_full);
        }
        // printf("here, result_count = %u\n", result_count);
        for (std::uint32_t i = 0; i < result_count; i++) {
            if (solutions.find(results[i]) == solutions.end()) {
                solutions.insert(results[i]);
                results_partial++;
                // std::fprintf(stderr, "New candidate sol: %" PRIu64 " %" PRIu64 "\n", results[i].lo, results[i].hi);
                Xoroshiro xr = results[i];
                full_check(xr, results_full);
            }
        }
        // exit(1);
        
        //if (r % 10 == 0) {
            // Xoroshiro next;
            // xSetSeed(&next, eqn_seed);
            // eqn_seed = xNextLongJ(&next);
        //}
        //else {
        //    eqn_seed += 1 + 81 + 81*81;
        //}
    }
}