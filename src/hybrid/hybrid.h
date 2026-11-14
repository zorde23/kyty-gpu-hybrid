#ifndef EMULATOR_SRC_HYBRID_HYBRID_H_
#define EMULATOR_SRC_HYBRID_HYBRID_H_

#include <cstdint>

// The AnyPS5 executable hosts Kyty's GPU as kyty_gpu.dll (the hybrid). The executable owns guest memory,
// threads, the exception handler and events; Kyty only tracks and draws. Aps5Host is the table of the
// executable's callbacks that kyty_gpu_init hands over (version 2).
struct Aps5Host {
	uint32_t version;
	// tracker: 0 = no tracking protection, 1 = read-only, 2 = no access; per 4 KiB page of donated memory.
	bool (*protect_host)(uint64_t address, uint64_t bytes, uint32_t tracker);
	// The executable's own export of a guest library NID (libkernel first), SysV ABI, or 0.
	uint64_t (*resolve)(const char* nid);
};

namespace Hybrid {

[[nodiscard]] bool            Enabled() noexcept;
[[nodiscard]] const Aps5Host* Host() noexcept;
void                          SetHost(const Aps5Host* host) noexcept;

// The executable's event queues (the guest's): Kyty's interrupt (kind 0: count in fflags, context in data,
// one pending event per context) and video out (kind 1) trigger semantics, built in the executable.
int AddEvent(int64_t eq, uint64_t ident, int16_t filter, void* udata, int kind, bool triggered, int64_t data);
int TriggerEvent(int64_t eq, uint64_t ident, int16_t filter, void* trigger_data);
int DeleteEvent(int64_t eq, uint64_t ident, int16_t filter);

// The executable's automated input (APS5_PAD_INPUT_FILE and what drives it) replaces the pad data while
// it runs; nothing changes otherwise.
void PadInjection(void* pad_data, uint64_t bytes);

// One of the executable's own helpers by name (KytyGpu*_nid_no_patch), or null outside the hybrid.
template <typename Function>
Function Executable(const char* name) noexcept {
	const auto* host = Host();
	return host == nullptr ? nullptr : reinterpret_cast<Function>(host->resolve(name));
}

// In the hybrid, the executable's implementation of a kernel NID instead of Kyty's own `function`.
template <typename Function>
Function Kernel(const char* nid, Function function) noexcept {
	const auto* host = Host();
	return host == nullptr ? function : reinterpret_cast<Function>(host->resolve(nid));
}

} // namespace Hybrid

#endif
