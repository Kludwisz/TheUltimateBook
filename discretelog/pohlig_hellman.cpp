#include <cstdint>
#include <cstdio>
#include <iostream>
#include "infint.h"
#include "bsgs.h"

static InfInt factors[9] = {
    InfInt(3) /*x 1*/,
    InfInt(5) /*x 1*/,
    InfInt(17) /*x 1*/,
    InfInt(257) /*x 1*/,
    InfInt(641) /*x 1*/,
    InfInt(65537) /*x 1*/,
    InfInt(274177) /*x 1*/,
    InfInt(6700417) /*x 1*/,
    InfInt(67280421310721) /*x 1*/,
};

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

void advance_infint(Xoroshiro *state, const InfInt& jump) {
    InfInt two64 = InfInt(1LL << 32) * InfInt(1LL << 32);
    uint64_t lo = (jump % two64).toUnsignedLongLong();
    uint64_t hi = (jump / two64).toUnsignedLongLong();

    FXTMatrix fxtm;
    fastXoroMatrixPower128(&XOROSHIRO_STANDARD_FXTM, hi, lo, &fxtm);
    fastAdvanceXoroshiroFXTM(state, &fxtm);
}

void xoroshiro_discrete_log_test(Xoroshiro a, Xoroshiro b) {
    FXTMatrix Ks, Kt, Ksinv, A;

    /*
    To perform baby-step-giant-step, we recover the underlying matrix that advances the Xoroshiro state 
    from a to b: A=M^n by stacking multiple state transforms s*M^i --A--> t*M^i (for i=0..127) 
    into a single big operation S * A = T. Having that equation with known S and T, it's easy to see how 
    A = S^-1 * T. Once we have the matrix A, the problem becomes a standard discrete logarithm in a matrix group.
    */
    Xoroshiro curr_s = a;
    Xoroshiro curr_t = b;
    FXTMatrix MT;
    transposeFXTM(&XOROSHIRO_STANDARD_FXTM, &MT);

    for (int row = 0; row < 128; row++) {
        int qj = row / 64;
        int j = row % 64;
        Ks.M[0][qj][j] = curr_s.lo;
        Ks.M[1][qj][j] = curr_s.hi;
        Kt.M[0][qj][j] = curr_t.lo;
        Kt.M[1][qj][j] = curr_t.hi;
        fastAdvanceXoroshiroFXTM(&curr_s, &MT);
        fastAdvanceXoroshiroFXTM(&curr_t, &MT);
    }

    // Reconstruct A = M^n
    invertFXTM(&Ks, &Ksinv);
    multiplyFXTM(&Ksinv, &Kt, &A);

    InfInt remainders[9];
    InfInt work_factors[9];
    for (int i = 0; i < 9; i++) {
        work_factors[i] = factors[i];
    }

    uint64_t n = 17;
    FXTMatrix Ad, Adinv;
    
    project_into_subgroup(&A, &Ad, n);
    invertFXTM(&Ad, &Adinv);
    uint64_t dl = solve_discrete_log(&Ad, &Adinv, n);
    printf("discrete log for subgroup N=%llu : %llu\n", n, dl);
    return;

    for (int i = 0; i < 9; i++) {
        uint64_t n = work_factors[i].toUnsignedLongLong();
        FXTMatrix Ad, Adinv;
        
        project_into_subgroup(&A, &Ad, n);
        invertFXTM(&Ad, &Adinv);
        uint64_t dl = solve_discrete_log(&Ad, &Adinv, n);

        remainders[i] = InfInt(dl);
        printf("discrete log for subgroup N=%llu : %llu\n", n, dl);
    }

    InfInt distance = get_value_from_remainders(work_factors, remainders, 9);
    std::cout << "distance = " << distance << '\n' << std::flush;

    {
        Xoroshiro t = a;
        advance_infint(&t, distance);
        printf("--- Test result\n");
        printf("got state (%016llx-%016llx)\n", t.hi, t.lo);
        printf("expected  (%016llx-%016llx)\n", b.hi, b.lo);
    }
}

int main() {
    //xoroshiro_discrete_log_test({1, 0}, {2, 0});

    Xoroshiro a = {1, 0};
    Xoroshiro b = a;
    xNextLong(&b);

    xoroshiro_discrete_log_test(a, b);
}

/*
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
*/