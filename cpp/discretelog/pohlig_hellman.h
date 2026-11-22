#ifndef POHLIG_HELLMAN_H
#define POHLIG_HELLMAN_H

#include "infint.h"
#include "../xoroshiro.h"

InfInt xoroshiro_distance(Xoroshiro a, Xoroshiro b, bool log_progress = false);
void advance_infint(Xoroshiro *state, const InfInt& jump);

#endif