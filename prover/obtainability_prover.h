#ifndef OBTAINABILITY_PROVER_H
#define OBTAINABILITY_PROVER_H

#include <vector>
#include "../xoroshiro.h"

struct ObtainabilityProof {
    // the initial state for which we start calculating the convergence point.
    Xoroshiro first_initial_state;
    // an integer such that no matter what the previous fishing sequence was, 
    // the sequence will fall into the first num_initial_states states starting 
    // at first_initial_state.
    int num_initial_states;

    // the xoroshiro state at which all possible initial states converge
    Xoroshiro convergence_state;

    // the required sequence of luck of the sea levels
    // in consecutive fishing catches starting at convergence_state.
    std::vector<int> lots_sequence;
};

bool get_obtainability_proof(Xoroshiro target_state, ObtainabilityProof& proof);

#endif