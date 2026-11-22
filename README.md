# Multi-enchantment books from fishing

Contains code Scriptline and I used to find a 21-enchantment book in Minecraft 1.21.4 (and probably above).
Uses an interesting technique which we call "Xoroshiro state prediction" to quickly search a large number of Xoroshiro states that are likely to produce the required sequence of `nextInt(50)` values.

Some of our findings:
- 21-enchantment book state: `hi = 0x9b1a153ed6ca3bb2, lo = 0x7f72e1e966dc96c5`
- 20-enchantment book state: `hi = 0x8ba2a3db099e8f65, lo = 0x1750ecc7f5c7fcec`

State distances of the fishing PRNG for the example seed "`Village at spawn`" (`-663035294`):
- 21-enchantment book: distance of `219997035380300381549483433782100004986` states
- 20-enchantment book: distance of  `1833540679362247710271556906392589490` states

Obtainability proofs generated with `cpp/prover`:

```
===== Good state - 21 enchants! lo=9183650982731224773 hi=-7270475288800511054
--- Item obtainability proof ---
Start state: (02a2389c22e24fd0-ddfde5d35dd7f417)
State range completness guaranteed after 12 states.
Convergence state: (d67d47d057087f71-f4c20827a2614d8c)
LOTS usage sequence: [
    LOTS_0 x 87
    LOTS_1 x 1
    LOTS_0 x 259
    LOTS_2 x 1
    LOTS_0 x 72
]
--------------------------------
Proof verification:
Expected (9b1a153ed6ca3bb2-7f72e1e966dc96c5)
Got      (9b1a153ed6ca3bb2-7f72e1e966dc96c5)
```

```
===== Good state - 20 enchants! lo=1680103004575366380 hi=-8384959395055890587
--- Item obtainability proof ---
Start state: (a60593b3228e51a3-1214574b32f2a74f)
State range completness guaranteed after 12 states.
Convergence state: (ac634b5e188bc604-68d273b496794926)
LOTS usage sequence: [
    LOTS_0 x 73
]
--------------------------------
Proof verification:
Expected (8ba2a3db099e8f65-1750ecc7f5c7fcec)
Got      (8ba2a3db099e8f65-1750ecc7f5c7fcec)
```

## Repo structure
`cpp/discretelog/` - discrete logarithm code that let us find state distances

`cpp/prover/` - verification and proof of in-game obtainability

`cuda/` - GPU kernel for finding good candidate states