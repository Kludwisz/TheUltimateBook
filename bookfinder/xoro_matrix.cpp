#include "xoro_matrix.h"
#include "utils.h"
#include <inttypes.h>

static const bool DEBUG_MODE = false;

// -----------------------------------------------------------

void copyFXTM(const FXTMatrix* from, FXTMatrix* to)
{
    for (int qi = 0; qi < 2; qi++)
    {
        for (int qj = 0; qj < 2; qj++)
        {
            for (int k = 0; k < 64; k++)
                (to->M)[qi][qj][k] = (from->M)[qi][qj][k];
        }
    }
}

void transposeFXTM(const FXTMatrix* matrix, FXTMatrix* transposed)
{
    for (int qi = 0; qi < 2; qi++) for (int qj = 0; qj < 2; qj++)
    {
        for (int j = 0; j < 64; j++)
        {
            const int sh = 63-j;
            uint64_t val = 0ULL;

            for (int i = 0; i < 64; i++)
            {
                val <<= 1;
                val |= ((matrix->M)[qi][qj][i] >> sh) & 1ULL;
            }
            (transposed->M)[qi][qj][j] = val;
        }
    }
}

// -----------------------------------------------------------

#ifdef __GNUC__
#include <x86intrin.h>
int getParity(uint64_t val) {
    return __builtin_parityll(val); // aka: GCC, do your thing!
}
#else
#include <intrin.h>
int getParity(uint64_t val) {
    return __popcnt64(val) & 1;
}
#endif

// bT is the FXTM "transposition" of b
void multiplyFXTM(const FXTMatrix* a, const FXTMatrix* bT, FXTMatrix* c)
{
    // calculate each bit in the result matrix separately
    for (int qi = 0; qi < 2; qi++) for (int qj = 0; qj < 2; qj++)
    {
        for (int i = 0; i < 64; i++)
        {
            uint64_t res = 0ULL;
            // these will be constant throughout the j-loop
            const uint64_t aLo = a->M[qi][0][i], aHi = a->M[qi][1][i];

            for (int j = 0; j < 64; j++)
            {
                res <<= 1;
                const uint64_t mulXor = (aLo & bT->M[0][qj][j]) ^ (aHi & bT->M[1][qj][j]);
                const int newBit = getParity(mulXor);
                res |= newBit; 
            }

            (c->M)[qi][qj][i] = res;
            // DEBUG("%llx\n", res);
        }
    }
}

void makeIdentityFXTM(FXTMatrix* fxtm) {
    // not exactly what you call "fast" but i was too lazy to hardcode an identity matrix lol
    fastXoroMatrixPower128(&XOROSHIRO_STANDARD_FXTM, FULL_64, FULL_64, fxtm);
}

void fastXoroMatrixPower(const FXTMatrix *matrix, uint64_t power, FXTMatrix *result)
{
    if (power == 0) {
        makeIdentityFXTM(result);
    }
    else {
        fastXoroMatrixPower128(matrix, 0ULL, power, result);
    }
}

// the two uint64_t fields give us the ability to advance by an arbitrary number
// of Xoroshiro states. TODO: consider adding a struct for the uint64_t's
void fastXoroMatrixPower128(const FXTMatrix* matrix, uint64_t powerUpper64, uint64_t powerLower64,  FXTMatrix* result)
{
    FXTMatrix res1 = { 0 }, res2 = { 0 };
    FXTMatrix pow1 = { 0 }, pow2 = { 0 };
    FXTMatrix transposed = { 0 };
    FXTMatrix *temp;
    
    bool isResultZero = true;

    FXTMatrix *currentPower = &pow1, *nextPower = &pow2, *currentResult = &res1, *nextResult = &res2;
    copyFXTM(matrix, currentPower);

    uint64_t power = powerLower64;
    for (int bitID = 0; bitID < 128; bitID++)
    {
        if (power == 0ULL)
        {
            if (bitID > 63)
                break;  // already did upper powers
            else if (powerUpper64 == 0ULL)
                break;  // there is no upper power, just break
            // else continue until condition 3. reached, filling all the lower powers
        }

        bool transpositionDone = false;
        if (power & 1ULL)
        {
            DEBUG("if power & 1:  power = %" PRIu64 "\n", power);
            if (isResultZero)
            {
                copyFXTM(currentPower, nextResult);
                DEBUG("copied successfully\n");
                isResultZero = false;
            }
            else
            {
                transposeFXTM(currentPower, &transposed);
                transpositionDone = true;
                DEBUG("else: transposed successfully\n");
                multiplyFXTM(currentResult, &transposed, nextResult);
                DEBUG("else: performed fast mul successfully\n");
            }
                
            // swap next and current result, clearing space for the next operations
            temp = currentResult;
            currentResult = nextResult;
            nextResult = temp;
        }

        // create the transposition of currentPower
        DEBUG("calc next:  power = %" PRIu64 "\n", power)
        if (!transpositionDone)
            transposeFXTM(currentPower, &transposed);

        // perform a very fast, quasi-quadratic matrix multiplication
        multiplyFXTM(currentPower, &transposed, nextPower);
        power >>= 1;
        DEBUG("Multiplied, current bitID = %d\n", bitID)

        // swap next and current power, making space for the next operations
        temp = currentPower;
        currentPower = nextPower;
        nextPower = temp;

        if (bitID == 63)
        {
            // at key point, switch out bits to upper power
            DEBUG("Switch to upper power\n")
            power = powerUpper64;
        }
    }

    // copy result to output
    copyFXTM(currentResult, result);

    // no dynamically allocated data, C will handle the
    // struct deallocations on its own
}

// ---------------------------------------

void advanceXoroshiroFXTM(Xoroshiro *state, const FXTMatrix* fxtm)
{
    uint64_t oldState[2] = { state->lo, state->hi };
    uint64_t newState[2] = { 0ULL, 0ULL };

    for (int qi = 0; qi < 2; qi++) for (int qj = 0; qj < 2; qj++)
    {
        for (int i = 0; i < 64; i++)
        {
            const int sh = 63 - i;

            for (int j = 0; j < 64; j++)
            {
                const uint64_t bit1 = ((fxtm->M)[qi][qj][i] >> j) & 1ULL;
                const uint64_t bit2 = (oldState[qi] >> sh) & 1ULL;
                newState[qj] ^= (bit1 & bit2) << j;
            }
        }
    }

    state->lo = newState[0];
    state->hi = newState[1];
}

void fastAdvanceXoroshiroFXTM(Xoroshiro *state, const FXTMatrix* fxtmT)
{
    uint64_t oldState[2] = { state->lo, state->hi };
    uint64_t newState[2] = { 0ULL, 0ULL };

    for (int qi = 0; qi < 2; qi++) {
        for (int i = 0; i < 64; i++) {
            const uint64_t mulXor = (oldState[0] & fxtmT->M[0][qi][63-i]) ^ (oldState[1] & fxtmT->M[1][qi][63-i]);
            newState[qi] |= static_cast<uint64_t>(getParity(mulXor)) << i;
        }
    }

    state->lo = newState[0];
    state->hi = newState[1];
}

static void xorRows(FXTMatrix *left, FXTMatrix *right, int src_q, int src_i, int dst_q, int dst_i)
{
    for (int qj = 0; qj < 2; qj++) {
        left->M[dst_q][qj][dst_i] ^= left->M[src_q][qj][src_i];
        right->M[dst_q][qj][dst_i] ^= right->M[src_q][qj][src_i];
    }
}

void invertFXTM(const FXTMatrix *fxtm, FXTMatrix *inverted)
{
    FXTMatrix left, right;
    copyFXTM(fxtm, &left);
    makeIdentityFXTM(&right);

    for (int diag = 0; diag < 128; diag++) {
        int p_q = diag / 64;
        int p_i = diag % 64;
        int p_bit = 63 - p_i;

        int val = (left.M[p_q][p_q][p_i] >> p_bit) & 1;
        if (val == 0) {
            // Find another row below the row currently being fixed that has a 1 in the target col
            for (int r = diag + 1; r < 128; r++) {
                int r_q = r / 64;
                int r_i = r % 64;
                int r_val = (left.M[r_q][p_q][r_i] >> p_bit) & 1;
                if (r_val == 1) {
                    xorRows(&left, &right, r_q, r_i, p_q, p_i);
                    break;
                }
            }
        }

        for (int row = 0; row < 128; row++) {
            if (row == diag) continue;
            int r_q = row / 64;
            int r_i = row % 64;
            int r_val = (left.M[r_q][p_q][r_i] >> p_bit) & 1;
            if (r_val == 1) {
                xorRows(&left, &right, p_q, p_i, r_q, r_i);
            }
        }
    }

    copyFXTM(&right, inverted);
}

/*
static void printXoro(const Xoroshiro* state)
{
    printf("Current Xoroshiro state:   %016llx-%016llx\n", state->lo, state->hi);
}

int main() {
    Xoroshiro a, b;
    a.lo = b.lo = 0;
    a.hi = b.hi = 1;

    FXTMatrix xInv, xInvInv;
    invertFXTM(&XOROSHIRO_STANDARD_FXTM, &xInv);
    invertFXTM(&xInv, &xInvInv);

    advanceXoroshiroFXTM(&a, &XOROSHIRO_STANDARD_FXTM);
    advanceXoroshiroFXTM(&b, &xInvInv);

    printXoro(&a);
    printXoro(&b);
}
*/