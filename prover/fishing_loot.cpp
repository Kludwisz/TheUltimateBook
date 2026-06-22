#include <cstdint>
#include <cmath>
#include "fishing_loot.h"
#include "../utils.h"
static const bool DEBUG_MODE = false;


// Enchant with levels utilities, code is mostly copied from
// https://github.com/minecraftathome/lootinator
// and so are the enchant with levels skip tables
enum EnchantableItem {
    FISHING_ROD = 0,
    BOW = 1,
    BOOK = 2
};
static int ENCHANT_WITH_LEVELS_GROUPS[3][60] = {
    { 0,1,1,1,1,2,2,2,2,2, 2,2,3,3,3,3,3,3,3,3, 5,5,5,5,5,6,6,6,6,6, 6,6,6,6,6,6,6 }, // fishing rod
    { 0,0,0,0,0,1,1,1,1,1, 1,1,1,1,1,3,3,3,3,3, 3,3,3,3,3,5,5,5,5,5, 5,5,5,5,5,5,5 }, // bow
    { 0,7,7,7,7,12,12,12,12,12, 16,16,19,19,19,23,23,23,23,23, 25,25,25,25,25,28,28,28,28,28, 27,27,27,27,27,27,27 }  // book
};

static void skip_enchant_with_levels_30(Xoroshiro *rand, EnchantableItem item)
{	
    const std::uint32_t enchantability = 1;
    std::int32_t level = 30;
	const std::int32_t delta = enchantability / 4 + 1;
	level += 1 + xNextInt(rand, delta) + xNextInt(rand, delta);
	const float amplifier = (xNextFloat(rand) + xNextFloat(rand) - 1.0F) * 0.15F;
	level = std::floor( ((float)level + (float)level * amplifier) + 0.5F );

	std::int32_t nGroups = ENCHANT_WITH_LEVELS_GROUPS[item][level];
	if (nGroups == 0) {
        return;
    }
	(void)xNextLong(rand);

	while (xNextInt(rand, 50) <= level) {
        DEBUG("added enchant\n");
        nGroups--;
		if (nGroups == 0) {
            break;
        }
        (void)xNextLong(rand);
		level /= 2;
	}
}

static void process_treasure(Xoroshiro *rand) 
{
    DEBUG("treasure\n");
    int item = xNextInt(rand, 6) - 2;
    if (item < 0 || item > 2) {
        return; // no extra stuff here
    }
    DEBUG("enchant with levels for item %d\n", item);
    skip_enchant_with_levels_30(rand, static_cast<EnchantableItem>(item));
}

static void process_junk(Xoroshiro *rand) 
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
        11, 11, 11, 11, 11, 11, 11, 11, 11, 11 // rotten flesh
    };

    int item = item_lookup[xNextInt(rand, 100)];
    if (item == 1 || item == 6) {
        xNextFloat(rand); // set_damage
    }
}

static void next_fishing_item(Xoroshiro *rand, int lots_level) 
{
    std::uint32_t wTreas = 5 + 2*lots_level;
    std::uint32_t wJunk = 10 - 2*lots_level;
    std::uint32_t wFish = 85 - lots_level;
    std::uint32_t wTotal = wTreas + wJunk + wFish;

    int w = xNextInt(rand, wTotal);
    DEBUG("w = %d", w);
    
    w -= wJunk;
    if (w < 0) {
        process_junk(rand);
        return;
    }

    w -= wTreas;
    if (w < 0) {
        process_treasure(rand);
        return;
    }

    // process fish
    xNextInt(rand, 100);
}

Xoroshiro get_next_state(Xoroshiro scx, int lots_level)
{
    next_fishing_item(&scx, lots_level);
    return scx;
}


// int main() {
//     Xoroshiro xr;
//     xSetFishingSeed(&xr, 395952196249720LL);
//     Xoroshiro xr2 = get_next_state(xr, 0);
// }
