# Shader recompiler: known gaps

## Subgroup-scope fences rely on AMD's wait counters

The fence after an atomic and the store completion (`S_WAITCNT_VSCNT`) are emitted as
`OpMemoryBarrier` with Subgroup scope (`EmitAtomicMemoryBarrier`, `EmitStoreCompletion`). Wider
scopes become an `s_barrier` on AMD's Windows driver, which hangs or reorders divergent code such
as a decoupled look-back.

Under the Vulkan memory model, a subgroup-scope release does not synchronize with another
workgroup's device-scope acquire. Ordering across workgroups therefore rests on the driver
emitting the wait counter for these fences, which holds on AMD (RX 6750 XT, Windows) and is not
guaranteed on NVIDIA, Intel or RADV.

- Check on another vendor: `lookback_subgroup_fences` (tools/cs_replay) and a radix sort replay.
- Diagnosis keys: `KYTY_ATOMIC_FENCE_SCOPE=buffer|legacy`, `KYTY_STORE_COMPLETION=legacy`.
- Fix when needed: pick the scope by vendor, as the profile already does for other features.

## glc FLAT and GLOBAL loads are still Volatile loads

A glc buffer load is a device-scope `OpAtomicLoad` (`LoadWordInBounds`), because AMD's Windows
driver hoists a Volatile load out of a loop that waits for another workgroup's flag. FLAT and
GLOBAL glc loads (`LoadBdaDword`, the PhysicalStorageBuffer path) still emit a Volatile `OpLoad`,
so a spin loop that reads its flag through a flat or global pointer can still hang on that driver.

- Fix when a title spins on a flat or global flag: emit the same atomic load in `LoadBdaDword` when
  the access is coherent.
