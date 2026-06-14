#include "state_solver.h"
#include "gaussian_elim.h"
#include "xoroshiro.h"
#include "xoro_matrix.h"
#include "utils.h"
const bool DEBUG_MODE = true;

#include "time.h"


void initParamSet(ParameterSet *obj, int size)
{
    obj->paramCount = size;
    obj->stateIDs = (int*)malloc(size * sizeof(int));
    obj->bitIDs = (int*)malloc(size * sizeof(int));
    obj->bitValues = (int*)malloc(size * sizeof(int));
}

void freeParamSet(ParameterSet *obj)
{
    free(obj->stateIDs);
    free(obj->bitIDs);
    free(obj->bitValues);
}

void initParamsFromFile(ParameterSet *obj, const char *filename)
{
    FILE* fptr = fopen(filename, "r");

    if (fptr == NULL)
    {
        DEBUG("File not found.\n")
        return;
    }
        
    // calc num of entries
    int entries = 0, temp;
    while (fscanf(fptr, "%d", &temp) > 0) 
        entries++;
    fclose(fptr);

    int equations = entries / 3;
    DEBUG("Equations in total: %d\n", equations)

    initParamSet(obj, equations);

    fptr = fopen(filename, "r");
    for (int i = 0; i < equations; i++)
    {
        fscanf(fptr, "%d", &(obj->stateIDs[i]));
        fscanf(fptr, "%d", &(obj->bitIDs[i]));
        fscanf(fptr, "%d", &(obj->bitValues[i]));
    }

    DEBUG("File input success!\n")
    fclose(fptr);
}

// ------------------------------------------------------------

int solveForStartingState(const ParameterSet *paramSet, Solution *sol)
{
    Equation* eqns = malloc(paramSet->paramCount * sizeof(Equation));

    // create all the equations
    FXTMatrix fxtm = { 0 }, fxtmTransposed = { 0 };

    for (int i = 0; i < paramSet->paramCount; i++)
    {
        int advanceCount = paramSet->stateIDs[i];
        fastXoroMatrixPower(&XOROSHIRO_STANDARD_FXTM, advanceCount, &fxtm);
        transposeFXTM(&fxtm, &fxtmTransposed);

        const int bid = paramSet->bitIDs[i] % 64; // unsure
        const int qj = paramSet->bitIDs[i] >= 64 ? 1 : 0;
        uint64_t lo = fxtmTransposed.M[0][qj][bid];
        uint64_t hi = fxtmTransposed.M[1][qj][bid];

        initEquation(&(eqns[i]), lo, hi, paramSet->bitValues[i]);
        // DEBUG("Equation:  %llu %llu  =  %d\n", eqns[i].lhs[0], eqns[i].lhs[1], eqns[i].rhs)
    }

    // use gaussian elimination to solve the system
    if (gaussianElimGF2(eqns, paramSet->paramCount, sol) != 0)
    {
        free(eqns);
        return -1;
    }

    free(eqns);
    return 0;
}

static int stateBitIndexOf(int varIndex) {
    //return varIndex;

    if (varIndex >= 64) {
        // high bits, reverse order
        return 63 - (varIndex - 64) + 64;
    }
    else {
        // low bits, reverse order
        return 63 - varIndex;
    }
}

// TODO make this much more efficient using parameter bitmasks
void getAllXoroshiroStates(const Solution* sol, Xoroshiro* states, uint32_t maxStates)
{
    int finalState[128];
    uint64_t maxParamVal = 1ULL << (sol->parameterCount);

    for (uint64_t paramValues = 0ULL; paramValues < maxParamVal && paramValues < maxStates; paramValues++)
    {
        // put the param values into the finalState array
        uint64_t pTemp = paramValues;
        for (int i = 127; i >= 0; i--)
        {
            if (!sol->isParameter[i]) {
                finalState[stateBitIndexOf(i)] = 0;
                continue;
            }
            
            finalState[stateBitIndexOf(i)] = pTemp & 1ULL;
            pTemp >>= 1;
        }

        // for each set parameter, flip all dependent variables which use the parameter
        for (int i = 127; i >= 0; i--) {
            if (!sol->isParameter[i] || finalState[stateBitIndexOf(i)] == 0) {
                continue;
            }
            
            int segment = i >= 64 ? 1 : 0;
            int bitIdx = 63 - (i % 64);
            uint64_t flag = 1ULL << bitIdx;

            for (int eID = 0; eID < sol->equationCount; eID++) {
                if (sol->equations[eID].lhs[segment] & flag) {
                    finalState[stateBitIndexOf(sol->dependentVar[eID])] ^= 1;
                }
            }
        }

        // RHS
        for (int eID = 0; eID < sol->equationCount; eID++) {
            if (sol->equations[eID].rhs == 1) {
                finalState[stateBitIndexOf(sol->dependentVar[eID])] ^= 1;
            }
        }

        // add the result
        uint64_t lo = 0ULL;
        uint64_t hi = 0ULL;
        for (int i = 63; i >= 0; i--)
        {
            lo <<= 1;
            hi <<= 1;
            lo |= finalState[i];
            hi |= finalState[64 + i];
        }

        states[paramValues].lo = lo;
        states[paramValues].hi = hi;
    }

    DEBUG("here\n");
}

// ------------------------------------------------------------

ParameterSet* genRandomParamSet(uint64_t seed, const Xoroshiro state)
{
    ParameterSet* ps = (ParameterSet*)malloc(sizeof(ParameterSet));

    int size = 120 + (rand() % 10);
    initParamSet(ps, size);

    Xoroshiro tempState = { 0ULL, 0ULL };

    for (int i = 0; i < size; i++)
    {
        tempState.lo = state.lo;
        tempState.hi = state.hi;

        int stateID = (rand() % 20) + 1;
        for (int j = 0; j < stateID; j++)
            xNextLong(&tempState);

        // choose random bit
        int bitID = rand() % 128;

        const uint64_t stateForBit = bitID >= 64 ? tempState.hi : tempState.lo;
        const int sh = 63 - (bitID % 64);
        const int bitValue = (stateForBit >> sh) & 1ULL;

        // DEBUG("Gen param set:  bitID = %d, stateForBit = %016llx, bitVal = %d\n", bitID, stateForBit, bitValue)

        ps->stateIDs[i] = stateID;
        ps->bitIDs[i] = bitID;
        ps->bitValues[i] = bitValue;
    }

    return ps;
}

int testSolverCorrectness(uint64_t seed)
{
    srand((int)seed);
    const Xoroshiro state = { rand() * rand() + 1, rand() * rand() + 1 };
    ParameterSet* ps = genRandomParamSet(seed, state);

    Solution sol;
    solveForStartingState(ps, &sol);

    if (sol.isContradictory)
    {
        DEBUG("Unexpected contradiction in equation system\n")
        freeParamSet(ps);
        free(ps);
        return 1;
    }

    int log2Sols = sol.parameterCount;
    if (log2Sols > 16)
    {
        DEBUG("Suspicious amount of solutions: log2(sols) = %d\n", log2Sols)
        freeParamSet(ps);
        free(ps);
        return 1;
    }

    uint64_t maxParamValue = 1ULL << log2Sols;
    Xoroshiro* possibleStates = (Xoroshiro*)malloc(maxParamValue * sizeof(Xoroshiro));
    
    getAllXoroshiroStates(&sol, possibleStates, 16);

    DEBUG("Got %llu possible solutions\n", maxParamValue)

    bool found = false;
    for (int i = 0; i < maxParamValue; i++)
        if (possibleStates[i].lo == state.lo && possibleStates[i].hi == state.hi)
            found = true;

    DEBUG("Checked all, found = %d\n", found)

    freeParamSet(ps);
    free(ps);
    free(possibleStates);

    return found ? 0 : 1;
}

int batchTestSolverCorrectness(int testCount)
{
    uint64_t startSeed = (uint64_t)time(NULL);

    for (int i = 0; i < testCount; i++)
    {
        if (testSolverCorrectness(startSeed + i) != 0)
        {
            DEBUG("Failed for seed %llu\n", startSeed + i)
            return 1;
        }
        DEBUG("Test %d OK!\n", i)
    }

    printf("Passed %d random tests.\n", testCount);
    return 0;
}
