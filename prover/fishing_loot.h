#ifndef FISHING_LOOT_H
#define FISHING_LOOT_H

#include "../xoroshiro.h"

// Xoroshiro generator with state counter
struct StateCountingXoroshiro {
    std::uint32_t counter = 0;
    Xoroshiro state;

    std::uint64_t nextLong();
    int nextInt(std::uint32_t n);
    float nextFloat();
};

StateCountingXoroshiro get_next_state(StateCountingXoroshiro scx, int lots_level);

#endif