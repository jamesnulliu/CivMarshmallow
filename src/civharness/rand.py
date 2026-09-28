"""Python port of Freeciv's RNG seeding (utility/rand.c, verified against
3.2.5 source).

Why this exists: a savegame's `[random]` section is loaded *before* its
`[settings]` (savegame3.c sg_load_random -> sg_load_settings order), so if
the stored stream is invalidated the server seeds from wall-clock entropy —
`set gameseed` can never influence a resumed game. Deterministic re-seeded
branching therefore requires writing the post-`fc_srand(seed)` generator
state directly into the save. This module computes that state exactly as the
engine would: v[0]=seed, v[i]=3*v[i-1]+257, indices (j,k,x)=(0,31,55), then
10,000 warm-up draws of fc_rand(MAX_UINT32).
"""

MAX32 = 0xFFFFFFFF


def fc_srand_state(seed: int) -> tuple[int, int, int, list[int]]:
    """Return (j, k, x, v[56]) — the generator state after fc_srand(seed)."""
    v = [0] * 56
    v[0] = seed & MAX32
    for i in range(1, 56):
        v[i] = (3 * v[i - 1] + 257) & MAX32
    j, k, x = 0, 31, 55
    # Warm-up: 10,000 draws of fc_rand(MAX_UINT32): divisor=1, so only the
    # value MAX_UINT32 itself is rejected and redrawn.
    for _ in range(10000):
        while True:
            new = (v[j] + v[k]) & MAX32
            x = (x + 1) % 56
            j = (j + 1) % 56
            k = (k + 1) % 56
            v[x] = new
            if new != MAX32:
                break
    return j, k, x, v


def random_section(seed: int) -> str:
    """The complete `[random]` savegame section for fc_srand(seed), matching
    sg_save_random's exact formatting ("%8x", space-padded lowercase hex)."""
    j, k, x, v = fc_srand_state(seed)
    lines = [
        "[random]",
        "saved=TRUE",
        f"index_J={j}",
        f"index_K={k}",
        f"index_X={x}",
    ]
    for row in range(8):
        vals = " ".join(f"{v[7 * row + i]:8x}" for i in range(7))
        lines.append(f'table{row}="{vals}"')
    return "\n".join(lines) + "\n"
