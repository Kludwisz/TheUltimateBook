#include <cstdint>
#include <cmath>
#include "fishing_loot.h"


// Xoroshiro with state counter
std::uint64_t StateCountingXoroshiro::nextLong() {
    counter++;
    return xNextLong(&state);
}

int StateCountingXoroshiro::nextInt(std::uint32_t n) {
    uint64_t r = (StateCountingXoroshiro::nextLong() & 0xFFFFFFFF) * n;
    if ((uint32_t)r < n)
    {
        while ((uint32_t)r < (~n + 1) % n)
        {
            r = (StateCountingXoroshiro::nextLong() & 0xFFFFFFFF) * n;
        }
    }
    return r >> 32;
}

float StateCountingXoroshiro::nextFloat() {
    return (StateCountingXoroshiro::nextLong() >> (64-24)) * 5.9604645E-8F;
}


// Enchant with levels utilities
enum EnchantableItem {
    FISHING_ROD = 0,
    BOW = 1,
    BOOK = 2
};
static int ENCHANT_WITH_LEVELS_GROUPS[3][60] = {
    { }, // TODO fishing rod
    { }, // TODO bow
    { }  // TODO book
};

static void skip_enchant_with_levels_30(StateCountingXoroshiro *scx, EnchantableItem item)
{
    // Courtesy of https://github.com/minecraftathome/lootinator :)
	
    const std::uint32_t enchantability = 1;
    std::int32_t level = 30;
	const std::int32_t delta = enchantability / 4 + 1;
	level += 1 + scx->nextInt(delta) + scx->nextInt(delta);
	const float amplifier = (scx->nextFloat() + scx->nextFloat() - 1.0F) * 0.15F;
	level = std::floor( ((float)level + (float)level * amplifier) + 0.5F );

	std::int32_t nGroups = ENCHANT_WITH_LEVELS_GROUPS[item][level];
	if (nGroups == 0) {
        return;
    }
	scx->nextLong();

	while (scx->nextInt(50) <= level) {
        nGroups--;
		if (nGroups == 0) {
            break;
        }
        scx->nextLong();
		level /= 2;
	}
}

static void process_treasure(StateCountingXoroshiro *scx) 
{
    int item = scx->nextInt(6) - 2;
    if (item < 0 || item > 2) {
        return; // no extra stuff here
    }
    skip_enchant_with_levels_30(scx, ENCHANT_WITH_LEVELS_GROUPS[item]);
}

static void process_junk(StateCountingXoroshiro *scx) 
{
    const int item_lookup[] = {
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, // boots - skip 1 extra
        2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
        3, 3, 3, 3, 3, 3, 3, 3, 3, 3,
        4, 4, 4, 4, 4, 4, 4, 4, 4, 4, // water bottle
        5, 5, 5, 5, 5, // string
        6, 6, // rod - skip 1 extra
        7, 7, 7, 7, 7, 7, 7, 7, 7, 7, // bowl
        8, 8, 8, 8, 8, // stick
        9, // ink sac
        10, 10, 10, 10, 10, 10, 10, 10, 10, 10,
        11, 11, 11, 11, 11, 11, 11, 11, 11, 11, // rotten flesh
    };

    int item = item_lookup[scx->nextInt(100)];
    if (item == 1 || item == 6) {
        scx->nextFloat(); // set_damage
    }
}

static void next_fishing_item(StateCountingXoroshiro *scx, int lots_level) 
{
    std::uint32_t wTreas = 5 + 2*lots_level;
    std::uint32_t wJunk = 10 - 2*lots_level;
    std::uint32_t wFish = 85 - lots_level;
    std::uint32_t wTotal = wTreas + wJunk + wFish;

    int w = scx->nextInt(wTotal);
    
    w -= wTreas;
    if (w < 0) {
        process_treasure(scx);
        return;
    }

    w -= wJunk;
    if (w < 0) {
        process_junk(scx);
        return;
    }

    // process fish
    scx->nextInt(100);
}

StateCountingXoroshiro get_next_state(StateCountingXoroshiro scx, int lots_level)
{
    next_fishing_item(&scx, lots_level);
    return scx;
}
