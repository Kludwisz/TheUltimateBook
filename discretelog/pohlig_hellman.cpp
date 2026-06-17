#include <cstdint>
#include <cstdio>
#include "infint.h"

InfInt factors[9] = {
    InfInt(67280421310721) /*x 1*/,
    InfInt(6700417) /*x 1*/,
    InfInt(274177) /*x 1*/,
    InfInt(65537) /*x 1*/,
    InfInt(641) /*x 1*/,
    InfInt(257) /*x 1*/,
    InfInt(17) /*x 1*/,
    InfInt(5) /*x 1*/,
    InfInt(3) /*x 1*/
};

// InfInt factors[9] = {
//     101,
//     23,
//     19,
//     17,
//     13,
//     7,
//     5,
//     3,
//     2
// };

InfInt extended_gcd_128(InfInt a, InfInt b, InfInt& s_out, InfInt& t_out) {
    InfInt old_r = a, r = b;
    InfInt old_s = InfInt(1), s = InfInt(0);
    InfInt old_t = InfInt(0), t = InfInt(1);
    
    while (r != 0) {
        InfInt quotient = old_r / r;
        InfInt new_r = old_r - quotient * r;
        InfInt new_s = old_s - quotient * s;
        InfInt new_t = old_t - quotient * t;
        old_r = r;
        old_s = s;
        old_t = t;
        r = new_r;
        s = new_s;
        t = new_t;
    }

    s_out = old_s;
    t_out = old_t;
    return old_r;
}

InfInt get_value_from_remainders(InfInt moduli[], InfInt remainders[], uint32_t num_eqns) {
    for (int32_t i = num_eqns - 2; i >= 0; i--) {
        InfInt m1, m2;
        InfInt n1 = moduli[i], n2 = moduli[i+1];
        InfInt a1 = remainders[i], a2 = remainders[i+1];
        (void)extended_gcd_128(n1, n2, m1, m2);
    
        InfInt mod_product = n1 * n2;
        InfInt val = (a1*m2*n2 + a2*m1*n1) % mod_product;
        if (val < 0) {
            val += mod_product;
        }

        remainders[i] = val;
        moduli[i] = n1 * n2;
    }
    return remainders[0];
}


int main() {
    const uint32_t num_eqns = 9;

    InfInt rems[9];
    for (int i = 0; i < 9; i++) {
        rems[i] = InfInt(8-i);
    }

    InfInt temp_factors[9];
    for (int i = 0; i < 9; i++) temp_factors[i] = factors[i];

    InfInt val = get_value_from_remainders(temp_factors, rems, num_eqns);
    for (int i = 0; i < num_eqns; i++) {
        std::cout << (val % factors[i]) << '\n';
    }

    return 0;
}