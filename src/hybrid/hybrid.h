#ifndef EMULATOR_SRC_HYBRID_HYBRID_H_
#define EMULATOR_SRC_HYBRID_HYBRID_H_

#include <cstdint>

// The AnyPS5 executable hosts Kyty's GPU as kyty_gpu.dll (the hybrid). The executable owns guest memory,
// threads, the exception handler and events; Kyty only tracks and draws. Aps5Host is the table of the
// executable's callbacks that kyty_gpu_init hands over (version 1).
struct Aps5Host {
	uint32_t version;
	// tracker: 0 = no tracking protection, 1 = read-only, 2 = no access; per 4 KiB page of donated memory.
	bool (*protect_host)(uint64_t address, uint64_t bytes, uint32_t tracker);
};

namespace Hybrid {

[[nodiscard]] bool            Enabled() noexcept;
[[nodiscard]] const Aps5Host* Host() noexcept;
void                          SetHost(const Aps5Host* host) noexcept;

} // namespace Hybrid

#endif
