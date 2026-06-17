#ifndef BSGS_H
#define BSGS_H

#include "../xoro_matrix.h"

uint64_t solve_discrete_log(FXTMatrix *A, FXTMatrix *Ainv, uint64_t n);
void project_into_subgroup(const FXTMatrix *A, FXTMatrix *Ad, uint64_t f);

#endif