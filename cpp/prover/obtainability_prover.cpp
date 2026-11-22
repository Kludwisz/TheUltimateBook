#include "obtainability_prover.h"
#include "fishing_loot.h"
#include "../xoro_matrix.h"
#include "../utils.h"

#include <unordered_map>
#include <fstream>
#include <vector>

static const bool DEBUG_MODE = false;


void ObtainabilityProof::print() const
{
    std::printf("--- Item obtainability proof ---\n");
    std::printf("Start state: (%016llx-%016llx)\n", first_initial_state.hi, first_initial_state.lo);
    std::printf("State range completness guaranteed after %d states.\n", num_initial_states);
    std::printf("Convergence state: (%016llx-%016llx)\n", convergence_state.hi, convergence_state.lo);
    std::printf("LOTS usage sequence: [\n");

    int level = lots_sequence[0];
    int num = 0;
    for (int el : lots_sequence) {
        if (el != level) {
            std::printf("    LOTS_%d x %d\n", level, num);
            num = 0;
        }
        level = el;
        num++;
    }
    if (num > 0) {
        std::printf("    LOTS_%d x %d\n", level, num);
    }

    std::printf("]\n");
    std::printf("--------------------------------\n");
}


static bool find_target_forward(ObtainabilityProof &proof, const Xoroshiro& target, int max_offset, int fingerprint_states) 
{
    std::unordered_map<Xoroshiro, Xoroshiro, XoroshiroHash> reachable_states;
    std::unordered_map<Xoroshiro, int, XoroshiroHash> reachable_state_levels;

    Xoroshiro current = proof.convergence_state;
    reachable_states[current] = current;
    reachable_state_levels[current] = -1;

    for (int i = 0; i < fingerprint_states; i++) {
        Xoroshiro next = get_next_state(current, 0);
        reachable_states[next] = current;
        reachable_state_levels[next] = 0;
        current = next;
    }

    bool found = false;
    for (int i = 0; i < max_offset && !found; i++) {
        if (reachable_states.find(current) == reachable_states.end()) {
            (void)xNextLong(&current);
            continue;
        }

        DEBUG("current offset = %d\n", i);
        for (int lots = 0; lots <= 3; lots++) {
            //DEBUG("lots = %d\n", lots);
            Xoroshiro next_state = get_next_state(current, lots);
            bool should_update = reachable_states.find(next_state) == reachable_states.end()
                || (reachable_states.find(next_state) != reachable_states.end() && reachable_state_levels[next_state] > lots);
            
            if (should_update) {
                reachable_states[next_state] = current;
                reachable_state_levels[next_state] = lots;

                if (next_state == target) {
                    DEBUG("Target found\n");
                    found = true;
                    break;
                }
            }
        }
        
        (void)xNextLong(&current);
    }

    if (!found) {
        return false;
    }

    // reconstruct the LOTS order
    std::vector<int> lots_reversed;
    int current_level = reachable_state_levels[target];
    Xoroshiro current_xoro = reachable_states[target];
    DEBUG("Reconstructing LOTS sequence\n");
    
    while (current_level != -1) {
        lots_reversed.push_back(current_level);
        current_level = reachable_state_levels[current_xoro];
        current_xoro = reachable_states[current_xoro];
    }
    DEBUG("while loop done\n");

    proof.lots_sequence = std::vector<int>();
    for (int i = lots_reversed.size() - 1, j = 0; i >= 0; i--, j++) {
        DEBUG("%d:  LOTS=%d\n", j, lots_reversed[i]);
        proof.lots_sequence.push_back(lots_reversed[i]);
    }

    DEBUG("All done\n");
    return true;
}

static void calculate_min_initial_states(ObtainabilityProof &proof)
{
    int cutoff = 36;
    proof.num_initial_states = 2 + 1 + 5; // treasure item + set_damage + enchant_with_levels

    Xoroshiro state = proof.first_initial_state;
    xSkipN(&state, proof.num_initial_states);

    while (true) {
        int v1 = xNextInt(&state, 50);
        int v2 = xNextInt(&state, 50);
        if (v1 > cutoff && v2 > cutoff) {
            break; // there is no way for an item to pass this check.
        }
        cutoff /= 2;
        proof.num_initial_states += 2;
    }

    DEBUG("min_initial = %d\n", proof.num_initial_states);
}

static Xoroshiro get_convergent_state(ObtainabilityProof &proof) 
{
    std::unordered_map<Xoroshiro, int, XoroshiroHash> reachable_states;

    int num_active_seqns = proof.num_initial_states;
    bool* active_seqns = new bool[num_active_seqns];

    Xoroshiro temp = proof.first_initial_state;
    for (int i = 0; i < num_active_seqns; i++) {
        active_seqns[i] = true;
        reachable_states[temp] = i;
        (void)xNextLong(&temp);
    }

    Xoroshiro current = proof.first_initial_state;
    int max_advance = 0;
    for (int i = 0; num_active_seqns > 1; i++) {
        if (reachable_states.find(current) == reachable_states.end()) {
            (void)xNextLong(&current);
            continue;
        }
        
        int current_seqn = reachable_states[current];
        Xoroshiro next = get_next_state(current, 0);

        if (reachable_states.find(next) == reachable_states.end()) {
            reachable_states[next] = current_seqn;
        }
        else {
            active_seqns[current_seqn] = false;
            num_active_seqns--;
            DEBUG("active=%d\n", num_active_seqns);
        }

        if (num_active_seqns == 1) {
            DEBUG("Converged\n", i);
            max_advance = i + 100;
            break;
        }

        (void)xNextLong(&current);
    }

    // find the last state of the active sequence
    int seqn_id = 0;
    for (; seqn_id < proof.num_initial_states; seqn_id++) {
        if (active_seqns[seqn_id]) {
            break;
        }
    }
    Xoroshiro probe = proof.first_initial_state;
    Xoroshiro conv_state;

    for (int i = 0; i < max_advance; i++) {
        if (reachable_states.find(probe) != reachable_states.end()) {
            if (seqn_id == reachable_states[probe]) {
                DEBUG("Convergent state at i=%d\n", i);
                conv_state = probe;
            }
        }
        (void)xNextLong(&probe);
    }

    delete[] active_seqns;
    return conv_state;
}


bool get_obtainability_proof(Xoroshiro target_state, ObtainabilityProof &proof, int max_offset, int fingerprint_states)
{
    const int state_offset = max_offset;
    FXTMatrix back, backT;
    fastXoroMatrixPower128(&XOROSHIRO_STANDARD_FXTM, FULL_64, FULL_64 - state_offset, &back);
    transposeFXTM(&back, &backT);
    proof.first_initial_state = target_state;
    fastAdvanceXoroshiroFXTM(&proof.first_initial_state, &backT);

    calculate_min_initial_states(proof);
    proof.convergence_state = get_convergent_state(proof);

    return find_target_forward(proof, target_state, max_offset, fingerprint_states);
}


static int test_success_rate() {
    int back_states = 1000;
    int total = 1000, success = 0;

    for (uint64_t i = 0; i < 1000; i++) {
        ObtainabilityProof proof;
        Xoroshiro state = {i, 1};
        bool got_proof = get_obtainability_proof(state, proof, back_states, 50);
        if (got_proof) success++;
    }

    printf("Success rate: %d/%d\n", success, total);
    return 0;
}

int main() {
    FILE* fptr = fopen("input.txt", "r");
    if (fptr == NULL) {
        std::printf("input.txt not found\n");
        return 1;
    }

    std::int64_t lo, hi;
    int enchants;
    while (fscanf(fptr, "lo = %" PRId64 ", hi = %" PRId64 ", enchants = %d\n", &lo, &hi, &enchants) == 3) {
        Xoroshiro target = {static_cast<uint64_t>(lo), static_cast<uint64_t>(hi)};
        Xoroshiro book_state = back_to_book_roll(target);
        Xoroshiro no_result = {0, 0};
        if (book_state == no_result) {
            continue;
        }

        int back_states = 150;
        ObtainabilityProof proof;
        if (get_obtainability_proof(book_state, proof, back_states, 50)) {
            std::printf("===== Good state - %d enchants! lo=%lld hi=%lld\n", enchants, book_state.lo, book_state.hi);
            proof.print();
        }
        else {
            continue;
        }

        Xoroshiro test = proof.convergence_state;
        for (int lots : proof.lots_sequence) {
            test = get_next_state(test, lots);
        }

        std::printf("Proof verification:\n");
        std::printf("Expected (%016llx-%016llx)\n", book_state.hi, book_state.lo);
        std::printf("Got      (%016llx-%016llx)\n\n", test.hi, test.lo);
    }    
}
