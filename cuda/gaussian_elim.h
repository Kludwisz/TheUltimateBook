#pragma once
#include "inttypes.h"
#include "stdbool.h"

typedef struct Equation Equation;
struct Equation {
    uint64_t lhs[2];
    int rhs;
};

typedef struct Solution Solution;
struct Solution {
    bool isContradictory;

    int knownVariableCount;
    int parameterCount;
    bool isParameter[128];

    int equationCount;
    Equation equations[128];
    int dependentVar[128];
};

// ----------------------------------

int stateBitIndexOf(int varIndex);
void initEquation(Equation *eq, uint64_t lhsLo, uint64_t lhsHi, int rhs);
void xorEquationBy(Equation *a, const Equation *b);
void printEquation(Equation *eq);
void printEquation(const Equation *eqn, int eq_index);

int gaussianElimGF2(Equation equations[], int equationCount, Solution* sol);