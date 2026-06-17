#include <cstdint>
#include <cstdio>
#include <unordered_map>
#include <functional>
#include <cmath>

#include "bsgs.h"
#include "infint.h"

uint64_t solve_discrete_log(FXTMatrix *A, FXTMatrix *Ainv, uint64_t n)
{
    const uint64_t m = std::ceil(std::sqrt(n));

    FXTMatrix AT, Ainv_m, Ainv_mT;
    transposeFXTM(A, &AT);
    fastXoroMatrixPower(Ainv, m, &Ainv_m);
    transposeFXTM(&Ainv_m, &Ainv_mT);

    // You might ask what Xoroshiro is doing here, since at this point it's simply
    // a discrete log problem between matrices. The idea here is that we can take some
    // "test state" and store the result of its multiplication by the matrices instead
    // of the matrices themselves. This saves us both from doing matrix multiplication
    // AND from storing the full matrices in the LUT.
    std::unordered_map<Xoroshiro, uint64_t, XoroshiroHash> baby_step;

    printf("baby step states:\n");
    Xoroshiro x0 = {1, 0};
    for (uint64_t j = 0; j < m; j++) {
        fastAdvanceXoroshiroFXTM(&x0, &AT);
        printf("%016llx-%016llx\n", x0.hi, x0.lo);
        baby_step[x0] = j;
    }

    printf("giant step states:\n");
    Xoroshiro x1 = {1, 0};
    for (uint64_t i = 0; i < m; i++) {
        fastAdvanceXoroshiroFXTM(&x1, &Ainv_mT);
        printf("%016llx-%016llx\n", x1.hi, x1.lo);
        if (baby_step.find(x1) != baby_step.end()) {
            return i * m + baby_step[x1];
        }
    }

    return 0;
}


void project_into_subgroup(const FXTMatrix *A, FXTMatrix *Ad, uint64_t f) {
    InfInt two64 = InfInt(1LL << 32) * InfInt(1LL << 32);

    InfInt bigF = InfInt(f);
    InfInt bigN = two64 * two64 - InfInt(1);
    InfInt bigD = bigN / bigF;

    uint64_t lo = static_cast<uint64_t>((bigD % two64).toUnsignedLongLong());
    uint64_t hi = static_cast<uint64_t>((bigD / two64).toUnsignedLongLong());
    printf("Projection: %016llx-%016llx\n", hi, lo);

    fastXoroMatrixPower128(A, hi, lo, Ad);
}
