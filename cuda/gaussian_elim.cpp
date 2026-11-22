#include "gaussian_elim.h"
#include "utils.h"
#include <stdio.h>
#include <stdlib.h>

static const bool DEBUG_MODE = false;

void printEquation(const Equation *eqn, int eq_index) {
    int set_bits[128];
    int count = 0;

    for (int bit = 0; bit < 128; bit++) {
        int word = bit / 64;
        int shift = 63 - (bit % 64);
        if (eqn->lhs[word] & (1ULL << shift)) {
            set_bits[count++] = bit;
        }
    }

    if (count == 0) {
        printf("0 = %d", eqn->rhs);
        if (eqn->rhs != 0) {
            printf("  *** CONTRADICTION ***");
        }
        printf("\n");
        return;
    }

    for (int i = 0; i < count; i++) {
        int bit = set_bits[i];
        if (bit < 64) {
            printf("lo[%2d]", 63 - bit);       
        } else {
            printf("hi[%2d]", 63 - (bit - 64));
        }
        if (i < count - 1) {
            printf(" ^ ");
        }
    }

    printf(" = %d\n", eqn->rhs);
}

void initEquation(Equation *eq, uint64_t lhsLo, uint64_t lhsHi, int rhs)
{
    eq->lhs[0] = lhsLo;
    eq->lhs[1] = lhsHi;
    eq->rhs = rhs;
}

void xorEquationBy(Equation *a, const Equation *b)
{
    a->lhs[0] ^=  b->lhs[0];
    a->lhs[1] ^=  b->lhs[1];
    a->rhs ^= b->rhs;
}

int eqnCompare(const void* a_void, const void* b_void)
{
    const Equation *a = (Equation*)a_void, *b = (Equation*)b_void;

    if (a->lhs[0] > b->lhs[0])
        return 1;
    else if (a->lhs[0] < b->lhs[0])
        return -1;
    
    if (a->lhs[1] > b->lhs[1])
        return 1;
    else if (a->lhs[1] < b->lhs[1])
        return -1;
    
    return 0;
}

int stateBitIndexOf(int varIndex) {
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

int gaussianElimGF2(Equation equations[], int equationCount, Solution* sol)
{
    bool* usedEqns = (bool*)malloc(sizeof(bool) * equationCount);
    for (int i = 0; i < equationCount; i++)
        usedEqns[i] = false;

    sol->parameterCount = 128;
    sol->knownVariableCount = 0;
    for (int i = 0; i < 128; i++)
        sol->isParameter[i] = true;

    for (int seg = 0; seg <= 1; seg++)
    {
        uint64_t mask = ONLY_TOP_BIT;
        for (int i = 0; i < 64; i++, mask >>= 1)
        {
            for (int eID = 0; eID < equationCount; eID++)
            {
                if (usedEqns[eID] || (equations[eID].lhs[seg] & mask) == 0ULL)
                    continue;
                usedEqns[eID] = true;
                sol->isParameter[i + seg*64] = false;
                (sol->parameterCount)--;
                (sol->knownVariableCount)++;

                for (int eID2 = 0; eID2 < equationCount; eID2++)
                {
                    if (eID == eID2 || (equations[eID2].lhs[seg] & mask) == 0ULL)
                        continue;
                    xorEquationBy(&(equations[eID2]), &(equations[eID]));
                }
                break;
            }
        }
    }
    free(usedEqns);

    // sort the equations in ascending order for readability
    qsort(equations, equationCount, sizeof(equations[0]), eqnCompare);

    DEBUG("Sorted!\n");
    for (int i = 0; i < equationCount; i++) {
        DEBUG("Equation %d:  %016" PRIx64 " %016" PRIx64 " = %d\n", i+1, equations[i].lhs[0], equations[i].lhs[1], equations[i].rhs);
    }
    for (int i = 0; i < 128; i++) {
        if (sol->isParameter[i]) {
            DEBUG("Variable %d is a parameter\n", i);
        }
    }


    // check if there are any redundant equations (0 = 0) and/or falsehoods (0 = 1)
    int firstNonRedundantEq = 0;

    for (int eID = 0; eID < equationCount; eID++)
    {
        if (equations[eID].lhs[0] != 0ULL || equations[eID].lhs[1] != 0ULL)
            break;
        firstNonRedundantEq++;

        if (equations[eID].rhs == 1)
        {
            sol->isContradictory = true;
            return -1;
        }
    }

    // fill the Solution struct
    sol->isContradictory = false;
    sol->equationCount = equationCount - firstNonRedundantEq;

    for (int eID = 0; eID < sol->equationCount; eID++) {
        sol->equations[eID] = equations[eID + firstNonRedundantEq];
        // find the dependent variable

        for (int seg = 0; seg <= 1; seg++) {
            uint64_t flag = 1;
            for (int varID = 63; varID >= 0; varID--, flag <<= 1) {
                int trueID = varID + seg * 64;
                if ((sol->equations[eID].lhs[seg] & flag) && !(sol->isParameter[trueID])) {
                    sol->dependentVar[eID] = trueID;
                    DEBUG("Eqaution %d : dependent var = %d\n", eID, trueID);
                }
            }
        }
    }
    
    return 0;
}
